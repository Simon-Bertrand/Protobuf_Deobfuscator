from typing import Counter
from proto_schema_parser import FieldCardinality

from protobuf_deobfuscator.ast import PEnum, PField, PMessage, POneOf
from protobuf_deobfuscator.cross_ref import cross_ref_score
from protobuf_deobfuscator.graphs import ProtoGraphs
from protobuf_deobfuscator.hungarian import HungarianPathGenerator
import pandas as pd

def cross_ref_sim(
    cand,
    obf_root,
    ref_tree_base,
    tree_base,
    REF_REGISTER,
    REGISTER,
    callstack,
    _cache,
):
    base_sim = cand.similarity_with(
        obf_root, REF_REGISTER, REGISTER, callstack, _cache
    ).score
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
    return score

def candidate_covers_subject(a: Counter, b: Counter) -> bool:
    intersection = sum((a & b).values())  # min per key
    union = sum((a | b).values())  # max per key
    return intersection / union if union > 0 else 0

def generate_bias(graphs : ProtoGraphs):
    proposals = Counter()
    for nodename, node_data in graphs.G.nodes(data=True):
        for refnodename, refnode_data in graphs.Gref.nodes(data=True):
            if type(node_data["ast"]) is not type(refnode_data["ast"]):
                continue
            subject = candidate_covers_subject(
                node_data["ast"].get_fast_signature_counter(),
                refnode_data["ast"].get_fast_signature_counter(),
            )
            if subject > 0.15:
                proposals.update([(refnodename, node_data["ast"].full_name)])

    fast_sign_cands = pd.DataFrame(0, index=graphs.Gref.nodes, columns=graphs.G.nodes)
    for accept, i in proposals.items():
        fast_sign_cands.loc[accept[0], accept[1]] += 1
    fast_sign_nullbias = fast_sign_cands.stack()[fast_sign_cands.stack() == 0].to_dict()
    obfs_root_enums = sorted(
        [
            node_data["ast"]
            for nodename, node_data in graphs.G.nodes(data=True)
            if isinstance(node_data["ast"], PEnum)
        ],
        key=lambda x: len(x),
        reverse=True,
    )
    ref_root_enums = sorted(
        [
            node_data["ast"]
            for nodename, node_data in graphs.Gref.nodes(data=True)
            if isinstance(node_data["ast"], PEnum)
        ],
        key=lambda x: len(x),
        reverse=True,
    )

    callstack, _cache = set(), {}
    tree_base = graphs.proto.flatten_alls()
    ref_tree_base = graphs.ref_proto.flatten_alls()
    hg_results = HungarianPathGenerator(
        lambda x, y: cross_ref_sim(
            x,
            y,
            ref_tree_base,
            tree_base,
            graphs.REF_REGISTER_MAP,
            graphs.REGISTER_MAP,
            callstack,
            _cache,
        )
    ).compute(ref_root_enums, obfs_root_enums)
    enums_bias = (
        pd.DataFrame(
            hg_results.S,
            index=[el.full_name for el in ref_root_enums],
            columns=[el.full_name for el in obfs_root_enums],
        )
        .stack()
        .reset_index()[["level_0", "level_1", 0]]
        .set_index(["level_0", "level_1"])
        .to_dict()
    )
    return  (fast_sign_nullbias | enums_bias)
