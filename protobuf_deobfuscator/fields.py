from ast import Dict
from typing import Any, NamedTuple
from protobuf_deobfuscator import graphs
from protobuf_deobfuscator.ast import PAlls, PEnum, PField, PMapField, PMessage, POneOf, get_struct_by_name
from protobuf_deobfuscator.graphs import ProtoGraphs
from protobuf_deobfuscator.hungarian import HungarianPathGenerator
import pandas as pd

class AssociatedAsts(NamedTuple):
    obfu: PAlls
    ref: PAlls
    graphs: ProtoGraphs

    def desobfuscate_fields(self, callstack=None, _cache=None) -> Dict[str, Any]:
        return self._desobfuscate_fields(
            self.ref, self.obfu, callstack=callstack, _cache=_cache
        )

        
    def _desobfuscate_fields(self, node_ast : POneOf | PMessage | PEnum, obf_node_ast : POneOf | PMessage | PEnum, callstack=None, prefix=None, _cache=None):
        # Initialize callstack and cache if not provided
        if callstack is None:
            callstack = set()
        if _cache is None:
            _cache = {}
        
        # Compute Hungarian mapping
        if not type(node_ast) is type(obf_node_ast):
            return None
        if isinstance(node_ast, PEnum):
            return obf_node_ast.name if prefix is None else f"{prefix}.{obf_node_ast.name}"

        self_fields, other_fields = node_ast.elements, obf_node_ast.elements
        hg = HungarianPathGenerator(
            lambda x, y: x.similarity_with(
                y, graphs.REF_REGISTER_MAP, graphs.REGISTER_MAP, callstack=callstack, _cache=_cache
            ).score
        )
        query_key = (node_ast.full_name, obf_node_ast.full_name)

        hg_results = hg.compute(self_fields, other_fields)

        fields_mapping = (
            pd.DataFrame(
                hg_results.P,
                index=[ el.full_name for el in self_fields],
                columns=[el.full_name for el in other_fields],
            )
            .stack()
            .reset_index()
            .rename(columns={"level_0": "ref", "level_1": "obf", 0: "score"})
            .query("score != 0")
            .filter(items=["ref", "obf"])
            .set_index("ref")["obf"]
        )
        if query_key in callstack:
            return obf_node_ast.name
        callstack.add(query_key)
        mapping_dict = fields_mapping.to_dict()
        for key in list(mapping_dict.keys()):
            ref_ast = next(el for el in self_fields if el.full_name ==key)
            obf_ast = next(el for el in other_fields if el.full_name ==mapping_dict[key])
            mapping_dict[ref_ast.name] = mapping_dict.pop(key)
            obf_path_name = obf_ast.name if prefix is None else f"{prefix}.{obf_ast.name}"
            
            # Handle PField types
            if isinstance(obf_ast, PField) and isinstance(ref_ast, PField):
                if not obf_ast.is_primitive() and not ref_ast.is_primitive():
                    referenced_obftype = get_struct_by_name(obf_ast.type, graphs.REGISTER_MAP)
                    referenced_reftype = get_struct_by_name(ref_ast.type, graphs.REF_REGISTER_MAP)
                    if isinstance(referenced_obftype, PMessage):
                        # For nested message types, we need to include the field name that references the message
                        # in the path. So we pass obf_path_name as prefix, so nested fields build paths like:
                        # parent_path.field_name.nested_field_name
                        # This gives us proper field access paths: fham.fhah.fhaf.fgxn.fgxf.fgxg
                        nested_mapping = self._desobfuscate_fields(referenced_reftype, referenced_obftype, callstack=callstack, prefix=obf_path_name, _cache=_cache)
                        mapping_dict[ref_ast.name] = nested_mapping
                    elif isinstance(referenced_obftype, PEnum):
                        # For enums, return the obfuscated field name directly (not the enum type name)
                        # since enums are accessed through the field
                        mapping_dict[ref_ast.name] = obf_path_name
                    else:
                        print("Unknown type 2:", type(referenced_obftype))
                else:
                    # Primitive field - just use the path name
                    mapping_dict[ref_ast.name] = obf_path_name
            # Handle PMapField types
            elif isinstance(obf_ast, PMapField) and isinstance(ref_ast, PMapField):
                # For map fields, just use the path name (could be extended to handle key/value types)
                mapping_dict[ref_ast.name] = obf_path_name
            # Handle POneOf types
            elif isinstance(obf_ast, POneOf) and isinstance(ref_ast, POneOf):
                # Recursively desobfuscate OneOf fields
                nested_mapping = self._desobfuscate_fields(ref_ast, obf_ast, callstack=callstack, prefix=obf_path_name, _cache=_cache)
                if nested_mapping is not None:
                    mapping_dict[ref_ast.name] = nested_mapping
                else:
                    mapping_dict[ref_ast.name] = obf_path_name
            # Handle nested PMessage types
            elif isinstance(obf_ast, PMessage) and isinstance(ref_ast, PMessage):
                pass
            # Handle PEnum types (when enum appears directly in elements)
            elif isinstance(obf_ast, PEnum) and isinstance(ref_ast, PEnum):
                # For enums, return the obfuscated field name path directly (not the enum type name)
                # since enums are accessed through the field
                mapping_dict[ref_ast.name] = obf_path_name
            else:
                # Unknown type combination - just use the path name
                print("Unknown type combination:", type(ref_ast), type(obf_ast))
                mapping_dict[ref_ast.name] = obf_path_name

        callstack.remove(query_key)
        return mapping_dict