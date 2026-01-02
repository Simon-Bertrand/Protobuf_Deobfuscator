
from collections import Counter
import heapq
from typing import Dict, Set, Tuple
from protobuf_deobfuscator.cross_ref import cross_ref_score



def bottom_up_mapping(
    graphs, log, min_score_threshold: float = 0.1, callstack=set(), bias={},TOP_K = 16
) -> Dict[Tuple[str, str], float]:
    """
    Bottom-up root mapping using top-K candidates and cross-reference Hungarian similarity.
    """
    REF_REGISTER, REGISTER = graphs.REF_REGISTER_MAP, graphs.REGISTER_MAP
    # Flattened lists of all objects
    tree_base = graphs.proto.flatten_alls()
    ref_tree_base = graphs.ref_proto.flatten_alls()

    # Sort flattened tree by depth descending
    tree = sorted(tree_base, key=lambda el: getattr(el, "depth", 0), reverse=True)
    ref_tree = ref_tree_base.copy()

    ignored: Set[str] = set()
    processed: Set[str] = set()
    tree_proposals = {}
    callstack = callstack
    _cache = bias

    while tree:
        tree.sort(
            key=lambda el: (
                getattr(el, "depth", 0),
                len(el.elements) if hasattr(el, "elements") else 0,
            ),
            reverse=True,
        )
        element = tree[0]

        if element.full_name in processed or element.full_name in ignored:
            tree = [el for el in tree if not el.full_name.startswith(element.full_name)]
            continue
        obf_root = element.return_root()

        # Step 2: find TOP_K candidates by local similarity
        root_candidates = [
            el
            for el in ref_tree
            if type(el) is type(obf_root)
            if el.depth == 0
            if el.full_name not in processed
        ]
        if not root_candidates:
            tree = [el for el in tree if not el.full_name.startswith(element.full_name)]
            continue

         # Step 3: compute cross-reference score for TOP_K candidates

        root_candidate_sims = (
            (
                root_el,
                root_el.similarity_with(
                    obf_root, REF_REGISTER, REGISTER, callstack, _cache
                ).score,
            )
            for root_el in root_candidates
        )
        top_candidates = heapq.nlargest(TOP_K, root_candidate_sims, key=lambda x: x[1])

        if not top_candidates:
            continue

        best_score = -1.0
        best_candidate = None
        for cand, base_sim in top_candidates:
            cross_score, w = cross_ref_score(
                cand,
                obf_root,
                ref_tree_base,
                tree_base,
                REF_REGISTER,
                REGISTER,
                callstack,
                _cache,
            )
            score = (base_sim + cross_score) / (1 + w)
            if score > best_score:
                best_score = score
                best_candidate = cand
        
        # Step 4: reject if below threshold
        if best_candidate is None or best_score < min_score_threshold:
            ignored.add(obf_root.full_name)
            tree = [
                el for el in tree if not el.full_name.startswith(obf_root.full_name)
            ]
            continue

        # Step 5: accept mapping
        if best_score > min_score_threshold:
            tree_proposals[(best_candidate.full_name, obf_root.full_name)] = best_score
            log.info(
                "Accepted mapping: %s <-> %s (score=%.3f)",
                best_candidate.full_name,
                obf_root.full_name,
                best_score,
            )
        else:
            log.info(
                "Rejected mapping: %s <-> %s (score=%.3f)",
                best_candidate.full_name,
                obf_root.full_name,
                best_score,
            )
        processed.add(best_candidate.full_name)
        processed.add(obf_root.full_name)
        # Step 6: remove all objects under matched root
        tree = [el for el in tree if not el.full_name.startswith(obf_root.full_name)]
        ref_tree = [
            el
            for el in ref_tree
            if not el.full_name.startswith(best_candidate.full_name)
        ]

    return tree_proposals