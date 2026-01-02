from collections import defaultdict
import random
from typing import Dict, List, Set
import numpy as np
from scipy.optimize import linear_sum_assignment
import heapq

from protobuf_deobfuscator.ast import PAlls,PField,PMapField,POneOf
from protobuf_deobfuscator.hungarian import HungarianPathGenerator


def find_references(id_map: str, tree: List[PAlls]) -> List[PAlls]:
    # id_map is full_name (e.g. package.v1.Color)
    # field.type might be 'Color' or 'package.v1.Color'
    def match_type(field_type, target_full_name):
        if field_type == target_full_name:
            return True
        if target_full_name.endswith("." + field_type):
            return True
        return False
    
    fields = [el for el in tree if isinstance(el, (PField)) if match_type(el.type, id_map)]
    
    return (
        fields
        + [
            el
            for el in tree
            if isinstance(el, (PMapField))
            if match_type(el.key_type, id_map) or match_type(el.value_type, id_map)
        ]
        + [
            el
            for el in tree
            if isinstance(el, (POneOf))
            if any(match_type(f.type, id_map) for f in el.elements)
        ]
    )


def cross_ref_score(
    candidate_root,
    obf_root,
    ref_tree_base,
    tree_base,
    REF_REGISTER,
    REGISTER,
    callstack,
    _cache,
):
    """
    Compute the sum of Hungarian assignment similarity for all children referencing the root.
    """
    ref_children = find_references(candidate_root.full_name, ref_tree_base)
    obf_children = find_references(obf_root.full_name, tree_base)

    if len(ref_children) == 0 and len(obf_children) == 0:
        return 0, 0

    ref_roots = [el.return_root() for el in ref_children if el is not None]
    obf_roots = [el.return_root() for el in obf_children if el is not None]

    ref_parents = [el.parent_ast for el in ref_children if el is not None]
    obf_parents = [el.parent_ast for el in obf_children if el is not None]

    hg_parent = HungarianPathGenerator(
        lambda x, y: x.similarity_with(
            y, REF_REGISTER, REGISTER, callstack, _cache
        ).score
    )
    hg_parent_results = hg_parent.compute(ref_parents, obf_parents)
    hg_root = HungarianPathGenerator(
        lambda x, y: x.similarity_with(
            y, REF_REGISTER, REGISTER, callstack, _cache
        ).score
    )
    hg_root_results = hg_root.compute(ref_roots, obf_roots)
    return (
        hg_parent_results.score + hg_root_results.score
    ), hg_root_results.weight + hg_parent_results.weight
