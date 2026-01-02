# Protobuf Deobfuscator

## 1. Project Overview & Rationale
**protobuf_deobfuscator** is a heuristic solver for the **Quadratic Assignment Problem (QAP)**, an NP-Hard optimization problem, tailored specifically for **restoring obfuscated Google Protobuf schemas**.

### The Core Problem
When a Protobuf schema is obfuscated, all semantic names are lost (e.g., `Player` becomes `Message_1`, `health` becomes `field_3`). However, the **structural topology** (field types, cardinalities, nesting, dependencies) remains largely intact.
Given:
1.  **$G_{obf}$**: The Obfuscated Graph (from the current obfuscated source code).
2.  **$G_{ref}$**: The Reference Graph (from a previous version which is not obfuscated).

The goal is to find an isomorphism (or near-isomorphism) mapping $f: V_{obf} \to V_{ref}$ that maximizes global similarity.

### The Real-World Challenge: "Fuzzy" Matching
In 99% of reverse engineering scenarios, $G_{obf}$ is not just a renamed clone of $G_{ref}$. It is a **mutated** version caused by software updates:
-   **Fields Added/Removed**: A message might gain a new field.
-   **Type Changes**: `optional int32` might become `repeated int32`.
-   **Refactoring**: Messages might be moved or split.
-   **Dummy fields**: A message/enum might have fields that are not used in the code which is introduced by the obfuscator.
- **Field shuffling**: The order of fields might be changed, therefore, all the similarity metrics must be permutation-invariant.

A strict Subgraph Isomorphism algorithm would fail immediately ($Nodes_A \neq Nodes_B$ or $Fields_A \neq Fields_B$).

**A Solution**: Employ smooth similarity scores which differs from exact similarity metrics (with values in $\{0,1\}$).
-   Treat similarity as a continuous gradient (0.0 to 1.0).
-   Use the **Hungarian Algorithm** to solve assignments optimally even when sets are of different sizes.
-   Prioritize **Deep Structural Anchors** (bottom-up) to propagate certainty from the leaves to the roots.
-   If an enum has 5 values and the candidate has 6, the similarity should be ~$0.83$, not $0.0$. This allows the solver to find the "best fit" even in the presence of version drift.

---

## 2. Architecture & File Structure

The codebase is modular, with each file governing a specific aspect of the solver logic.

| File | Purpose | Key Property Leveraged |
| :--- | :--- | :--- |
| `ast.py` | **Custom AST Wrapper** | Extends `proto-schema-parser` nodes with similarity computation logic, caching, and cycle detection. |
| `graphs.py` | **Graph Lifting** | Wrapper around `ast.py` logic. Converts the AST into a NetworkX `DiGraph` to enable efficient parent/child traversals for Cross-Reference analysis. |
| `bottom_up.py` | **Main Solver Loop** | Implements the **Deepest-Leaf-First** strategy. It sorts the entire schema by dependency depth and matches the deepest roots first. |
| `biasing.py` | **Heuristic Anchoring** | Injects "Ground Truths" by spotting statistically unique signatures (e.g., "The only message with 5 ints and 2 strings") and specific Enum structures. |
| `cross_ref.py` | **Extrinsic Scoring** | Computes similarity based on **Context** ("Who uses me?"). Essential for distinguishing generic structures like `Vector3` vs `Color`. |
| `similarities.py` | **Math Kernels** | Contains the fuzzy comparison logic (smooth step functions, weighted sums) and penalty formulas for size mismatches. |
| `hungarian.py` | **Bipartite Matcher** | A wrapper around `scipy.optimize.linear_sum_assignment` that handles non-square matrices and normalized scoring. |
| `pipeline.py` | **Orchestrator** | High-level entry point that loads files, initializes graphs, generates bias, and runs the solver. |

---

## 3. The Algorithm: A Deep Dive

The process is broken down into four distinct phases.

### Phase 1: Ingestion & Lifting
1.  **Parsing**: The `.proto` files are parsed into a raw AST using `proto-schema-parser`.
2.  **Wrappers (`ast.py`)**: Raw nodes are wrapped in `PMessage`, `PEnum`, `PField` classes. These wrappers attach:
    -   `depth`: The distance to the furthest leaf in the dependency tree.
    -   `full_name`: The qualified unique name (`package.Msg.SubMsg`).
3.  **Graphing (`graphs.py`)**: A directed dependency graph is built.
    -   **Node Representation**: Every `Message` and `Enum` becomes a node.
    -   **Edge Representation**: Dependencies form edges. If `MessageA` contains a field of type `MessageB`, we draw $A \to B$.
    -   **Why**: This allows efficient traversal for *Cross-Reference* analysis ("Who uses me?") which is the inverse of the standard AST traversal ("Who do I use?").

### Phase 2: Heuristic Biasing (`biasing.py`)
Before the expensive recursive solver starts, we "cheat" by finding obvious matches.
1.  **Signature Matching**: We generate a `Counter` signature for every message (e.g., `{int32: 5, string: 2}`).
    -   **Negative Biasing (Pruning)**: Instead of explicitly linking unique signatures, we use them to **prune** impossible matches.
    -   We compare signatures using a **Jaccard Index**. If the overlap is too low (< 15%), we pre-fill the cache with `0.0`, effectively removing these candidates from consideration and speeding up the solver.
2.  **Enum Anchoring**: Enums are rarely unique structurally (just lists of ints). We explicitly run a specialized **Cross-Reference Pass** on all global Enums.
    -   We check: "Does the set of messages using `Enum_Obf` look like the set of messages using `Enum_Ref`?"
    -   Matches computed here are added to `_cache` with score 1.0.
3.  **Bias Injection**: High-confidence and manually specified matches are added to the `_cache` as **biases**. For example, if you have a function that matches two nodes based on custom criterias, you can inject that match into the cache with a high score (>1.0), this score will be propagated up the tree via the recursive similarity and cross-reference similarity scores. This effectively reduces the search space and improves the solver's performance, only if your manually specified matches are correct.

### Phase 3: The Bottom-Up Solver (`bottom_up.py`)
This is the core innovation. We do not try to match the "Root" (Main Schema Message) directly. That tree is typically too huge and QAP is NP-Hard.
Instead, we prune the problem from the bottom.

1.  **Queue Construction**: We sort all nodes by recursive depth *and then* by complexity (number of fields). We solve the deepest, most complex Roots first.
    -   **Why**:
        1.  **Reliability**: Matching a deep root implicitly validates the entire sub-tree. The recursive similarity score applied on root nodes is statistically more reliable with higher depth.
        2.  **Propagation**: Solving a deep root populates the cache for high-level containers, converting "unknown" fields into "known" anchors.
2.  **Iterative Matching**:
    For each target node $T \in Queue$:

    a.  **Filter**: Select candidates $C \in G_{ref}$ where `Depth(C) ≈ Depth(T)`.

    b.  **Fast Candidate Selection (TOP-K)**:
        <ul>
            <li>Compute Intrinsic Score only (Structure) for all valid candidates.</li>
            <li>Select the Top-K candidates with the highest structural similarity.</li>
            <li>Why is this Critical?: Calculating the Extrinsic Score (Context/Parent matching) is extremely expensive. Running it on 1000 candidates would freeze the solver. We filter the list to the K most structurally similar nodes first, then do the heavy context verification only on them.</li>
        </ul>

    c.  **Full Scoring**: For the Top-K survivors, compute $Score(T, C) = S_{intrinsic} + S_{extrinsic}$.

    d.  **Threshold**: If $Max(Score) > minThreshold$, we declare a match.

    e.  **Lock & Propagate (Greedy)**: The pair $(T, C)$ is added to `_cache` and removed from future consideration.
        -   **Greedy Approach**: Once a match is locked, it is **final**. We do not backtrack. This requires high confidence thresholds to prevent early "cache pollution" from false positives. This threshold could be dynamic to improve performance.
        -   **Crucial**: Now, when we calculate the score for a *Parent* of $T$, the recursion hits the cache for $T$ instead of recalculating, treating it as a solved anchor.

    f. **Memory Safety**: To prevent stack overflows on recursive Protobuf definitions (e.g., `Node -> Node`), we track the `callstack`. If a cycle is detected, the recursion is broken immediately, returning a heuristic score based on size ratio which could also be further improved.

---

## 4. Detailed Similarity Formulas

We use specific mathematical formulas to handle the "Fuzzy" matching requirements.

### 4.1 Intrinsic Similarity (Structure)
**Primitive Fields**:
$$ Sim(f_O, f_R) = \mathbb{I}(Type_O == Type_R) \times \mathbb{I}(Card_O == Card_R) $$
*Strict equality required. We never map `int` to `string`.*

**Enums (Smooth Length Decay)**:

Instead of $\frac{\min(len(E_A), len(E_B))}{\max(len(E_A), len(E_B))}$, we use:
$$ Sim(E_A, E_B) = \max \left( 0, 1.0 - \frac{|len(E_A) - len(E_B)|}{\min(len(E_A), len(E_B))} \right) $$
*Allows for minor additions/removals of enum values.*

**Messages (Recursive Hungarian)**:
To match Message A vs Message B:
1.  Construct Cost Matrix $M$ where $M_{ij} = Sim(FieldA_i, FieldB_j)$.
    -   If fields are custom types (Messages), recursively call $Sim(TypeA, TypeB)$.
    -   **Cycle Handling**: If recursion loops ($A \to B \to A$), return heuristic $\frac{Size_A}{Size_B}$.
2.  Solve Assignment: $S_{hungarian} = \text{LinearSumAssignment}(M)$.
3.  Compute Normalized Score:
    $$ S_{final} = \frac{\sum_{assigned} S_{hungarian}}{\max(Count_A, Count_B)} $$
*This effectively penalizes size mismatches linearly. A message with 20 fields matching a message with 2 fields will have a maximum score of $2/20 = 0.1$.*

**Maps (`PMapField`)**:
A map `map<K, V>` is treated as a tuple.
$$ Sim(Map_O, Map_R) = \frac{Sim(K_O, K_R) + Sim(V_O, V_R)}{2} $$

**OneOfs (`POneOf`)**:
Treated as a message where all fields are optional. We use the **Hungarian Algorithm** to align the fields inside the OneOf block, multiplied by the **SizeMatch** penalty for structural drift.
$$ Sim(Union_O, Union_R) = \text{HungarianCost}(Fields_O, Fields_R) \times SizeMatch(Fields_O, Fields_R) $$

### 4.2 Extrinsic Similarity (Context)
To distinguish identical structures (e.g., `Vector3 {x,y,z}` used in `Player` vs `Enemy`), we look at the parents.
$$ S_{extrinsic}(A, B) = \text{Hungarian}(Parents(A), Parents(B)) $$
*This answers: "Are the objects that use A similar to the objects that use B?"*

### 4.3 Total Score
The two scores are fused with a dynamic weight:
$$ S_{total} = \frac{S_{intrinsic} + W \cdot S_{extrinsic}}{1 + W} $$
*Typically, extrinsic weight $W$ is zero when no cross-reference is available, meaning only the structural similarity is considered.*

---

## 5. Dependencies

The project relies on a standard Data Science stack for optimization:

-   `numpy`: Matrix operations.
-   `scipy`: The `linear_sum_assignment` (Hungarian Algorithm) solver.
-   `pandas`: DataFrame management to keep index and column names for matrices.
-   `networkx`: Graph topology and traversal.
-   `proto-schema-parser`: Parsing raw `.proto` files into Python objects.

## 6. Installation

Since this project is not published on PyPI, you must clone the repository and install it locally using `pip`:

```bash
 pip install https://github.com/Simon-Bertrand/Protobuf_Desobfuscator/archive/main.zip
```

After installation, the CLI tool `protobuf_deobf` will be available in your path.

## 7. Usage

### Running the Solver (CLI)

The easiest way to use the tool is via the command line interface:

```bash
# With advanced tuning parameters
# -k 5: Check top 5 candidates (better accuracy, slower)
# -t 0.6: Require 60% similarity to accept a match
protobuf_deobf example/obf.proto example/ref.proto -k 5 -t 0.6
```
```text
Found 22 mappings:
game.v1.NPC -> game.v2.NPCOBF (Score: 0.696)
game.v1.WorldState -> game.v2.WorldStateOBF (Score: 0.650)
game.v1.Party -> game.v2.PartyOBF (Score: 0.618)
game.v1.Item -> game.v2.ItemOBF (Score: 0.713)
game.v1.QuestObjective -> game.v2.QuestObjectiveOBF (Score: 0.719)
game.v1.Vector3 -> game.v2.Vector3OBF (Score: 0.633)
game.v1.Quaternion -> game.v2.QuaternionOBF (Score: 0.750)
game.v1.Color -> game.v2.ColorOBF (Score: 0.697)
game.v1.Transform -> game.v2.TransformOBF (Score: 0.677)
game.v1.WeaponDetails -> game.v2.WeaponDetailsOBF (Score: 0.660)
game.v1.ArmorDetails -> game.v2.ArmorDetailsOBF (Score: 0.660)
game.v1.ConsumableDetails -> game.v2.ConsumableDetailsOBF (Score: 0.771)
game.v1.Inventory -> game.v2.InventoryOBF (Score: 0.697)
game.v1.Skill -> game.v2.SkillOBF (Score: 0.713)
game.v1.CharacterStats -> game.v2.CharacterStatsOBF (Score: 0.726)
game.v1.Player -> game.v2.PlayerOBF (Score: 0.657)
game.v1.ServerConfig -> game.v2.ServerConfigOBF (Score: 0.800)
game.v1.Quest -> game.v2.QuestOBF (Score: 0.714)
game.v1.SocialProfile -> game.v2.SocialProfileOBF (Score: 0.655)
game.v1.Missions -> game.v2.MissionsOBF (Score: 0.741)
game.v1.Faction -> game.v2.FactionOBF (Score: 0.820)
game.v1.TargetType -> game.v2.TargetTypeOBF (Score: 0.685)
```
The "k" argument specify the number of candidates to check for each node. Bigger "k" will take longer to desobfuscate but will provide better results. The "t" argument specify the minimum similarity required to accept a match and is heavily related to the number of software updates and graph differences. A smaller "t" will be required if the reference protobuf schemas are much older than the obfuscated schemas but it can also produce more false positives which will be propagated.

JSON outputs can also be asked:
```bash
# Output as JSON for easy parsing by other tools
protobuf_deobf example/obf.proto example/ref.proto -k 5 -t 0.6 --format json > mapping.json
```

Finally, to obtain the complete field mapping, the Hungarian algorithm is applied again at the field level with an exact similarity metric for the already matched message types.

However, if a given type appears multiple times, the resulting field mapping becomes ambiguous.

For example, consider the following matched message types:


```proto
message A {
  int32 fieldA = 1;
  int32 fieldB = 2;
}

message AOBF {
  int32 fieldC = 1;
  int32 fieldD = 2;
}

```
In this situation, with current information context, it is impossible to determine whether fieldA maps to fieldC or to fieldD. Resolving this ambiguity requires external information that is not available in the .proto files themselves. 

Consequently, reverse-engineering techniques must be employed: static analysis (e.g., using the Angr framework, IDA or Ghidra) and dynamic analysis (e.g., with Frida) provide viable approaches to disambiguate the field mapping in such cases.


---
 
## 8. Motivation

Reverse-engineering Protobuf-based protocols constitutes an intellectually stimulating mathematical challenge.

I developed this tool as an entry point into the field of reverse engineering. Through this work, I explored concepts from graph theory, encountered a concrete instance of an NP-hard problem, examined the usefulness of the Hungarian algorithm, particularly its limitations with non-square matrices and studied the mathematical formulation of the Quadratic Assignment Problem (QAP), where the primary objective is to recover the correct permutation matrix aligning two adjacency matrices. This project is strictly educational in nature and is not intended to be used in any unauthorized manner.

I also discovered CVXPY, a Python library for convex optimization, which I found to be a ergonomic and powerful tool for solving convex relaxations of the Quadratic Assignment Problem around an initial solution. However, in practice, the current greedy implementation produced significantly more accurate results than those obtained using CVXPY with the convex relaxation.