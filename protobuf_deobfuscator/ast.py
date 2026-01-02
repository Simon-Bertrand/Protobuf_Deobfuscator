from abc import ABC, abstractmethod
from pathlib import Path
from proto_schema_parser import Parser, ast
from typing import Any, Counter, List, NamedTuple, Optional
import pandas as pd
from protobuf_deobfuscator.hungarian import HungarianPathGenerator
from protobuf_deobfuscator.similarities import ExactSimilarityDefinition, SimilarityOutput, weighted_sum
import numpy as np

PROTOBUF_TYPES = {
    "int32",
    "int64",
    "uint32",
    "uint64",
    "sint32",
    "sint64",
    "fixed32",
    "fixed64",
    "sfixed32",
    "sfixed64",
    "float",
    "double",
    "bool",
    "string",
    "bytes",
    "google.protobuf.Any",
}



class SplittedFields(NamedTuple):
    fields_primit: List["PField"]
    field_custom: List["PField"]
    mapfields_primit: List["PMapField"]
    mapfields_custom: List["PMapField"]
    oneof_primit: List[List["PField"]]
    oneof_custom: List[List["PField"]]
    messages: List["PMessage"]
    enums: List["PEnum"]


class TypeSignature(NamedTuple):
    left_type: str
    right_type: Optional[str] = None
    cardinality: Optional[ast.FieldCardinality] = None


def set_fullname(parent: Optional[str], name: str) -> str:
    return (
        (parent + "." if len(parent) > 0 else "") + name if parent is not None else name
    )


def find_name(name, iterable):
    return next(filter(lambda x: x.name == name, iterable))


class PSharedMethods:
    def return_root(self):
        cand = self
        while (up := cand.parent_ast) is not None:
            cand = up
        return cand


class PHasSignature:
    def __init__(self, left_type=None, right_type=None, cardinality=None):
        self.left_type = left_type
        self.right_type = right_type
        self.cardinality = cardinality

    def get_signature(self) -> TypeSignature:
        return TypeSignature(
            left_type=self.left_type,
            right_type=self.right_type,
            cardinality=self.cardinality,
        )


class PHasPrimitiveCheck(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def is_primitive(self) -> bool:
        pass


class PHasSimilarity(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def similarity_with(self, other: Any, *_, **__) -> SimilarityOutput:
        pass


class PField(ast.Field, PSharedMethods, PHasSignature, PHasPrimitiveCheck):
    def __repr__(self) -> str:
        return f"PField(fn={self.full_name} name={self.name}, type={self.type}, cardinality={self.cardinality} d={self.depth})"

    def __init__(
        self,
        depth: int,
        parent: Optional[str],
        name: str,
        number: int,
        type: str,
        cardinality: Optional[ast.FieldCardinality] = None,
        options: List[ast.Option] = [],
        parent_ast: Optional[ast.Message] = None,
    ):
        self.full_name = set_fullname(parent, name)
        self.depth = depth
        ast.Field.__init__(
            self,
            name,
            number,
            type,
            cardinality,
            options,
        )
        PHasSignature.__init__(
            self,
            left_type=type,
            cardinality=cardinality,
        )
        self.parent_ast = parent_ast

    def is_primitive(self) -> bool:
        return self.type in PROTOBUF_TYPES

    def similarity_with(
        self,
        other: "PField",
        REGISTER1,
        REGISTER2,
        callstack: Optional[set] = None,
        _cache: Optional[dict] = None,
    ) -> SimilarityOutput:
        if not isinstance(other, PField):
            return SimilarityOutput.hard_penalization()
        match (self.is_primitive(), other.is_primitive()):
            case (True, True):
                return SimilarityOutput(
                    score=1.0 * (self.get_signature() == other.get_signature()),
                    weight=1,
                )
            case (False, False):
                return get_struct_by_name(self.type, REGISTER1).similarity_with(
                    get_struct_by_name(other.type, REGISTER2),
                    REGISTER1,
                    REGISTER2,
                    callstack,
                    _cache,
                )

        return SimilarityOutput.zero()


class PMapField(
    ast.MapField, PSharedMethods, PHasSignature, PHasPrimitiveCheck, PHasSimilarity
):
    def __repr__(self) -> str:
        return f"PMapField(fn={self.full_name} name={self.name}, key_type={self.key_type}, value_type={self.value_type} d={self.depth})"

    def __init__(
        self,
        depth,
        parent: Optional[str],
        name: str,
        number: int,
        key_type: str,
        value_type: str,
        options: List[ast.Option] = [],
        parent_ast: Optional[ast.Message] = None,
    ):
        self.full_name = set_fullname(parent, name)
        self.depth = depth
        ast.MapField.__init__(
            self,
            name,
            number,
            key_type,
            value_type,
            options,
        )
        PHasSignature.__init__(
            self,
            left_type=key_type,
            right_type=value_type,
        )
        self.parent_ast = parent_ast

    def key_is_primitive(self) -> bool:
        return self.key_type in PROTOBUF_TYPES

    def value_is_primitive(self) -> bool:
        return self.value_type in PROTOBUF_TYPES

    def is_primitive(self) -> bool:
        return self.value_is_primitive() or self.key_is_primitive

    def similarity_with(
        self,
        other: "PMapField",
        REGISTER1: dict,
        REGISTER2: dict,
        callstack: Optional[set] = None,
        _cache: Optional[dict] = None,
    ) -> SimilarityOutput:
        if not isinstance(other, PMapField):
            return SimilarityOutput.hard_penalization()
        scores, weights = [0, 0], [0, 0]
        match (self.key_is_primitive(), other.key_is_primitive()):
            case (True, True):
                scores[0] = 1.0 * (self.key_type == other.key_type)
                weights[0] = 1
            case (False, False):
                self_key_obj, other_key_obj = (
                    get_struct_by_name(self.key_type, REGISTER1),
                    get_struct_by_name(other.key_type, REGISTER2),
                )
                s = self_key_obj.similarity_with(
                    other_key_obj,
                    REGISTER1,
                    REGISTER2,
                    callstack=callstack,
                    _cache=_cache,
                )
                scores[0] = s.score
                weights[0] = s.weight

        match (self.value_is_primitive(), other.value_is_primitive()):
            case (True, True):
                scores[1] = 1.0 * (self.value_type == other.value_type)
                weights[1] = 1
            case (False, False):
                self_val_obj, other_val_obj = (
                    get_struct_by_name(self.value_type, REGISTER1),
                    get_struct_by_name(other.value_type, REGISTER2),
                )
                s = self_val_obj.similarity_with(
                    other_val_obj,
                    REGISTER1,
                    REGISTER2,
                    callstack=callstack,
                    _cache=_cache,
                )
                scores[1] = s.score
                weights[1] = s.weight
        total_weight = sum(weights)
        if total_weight == 0:
            return SimilarityOutput.null()
        return SimilarityOutput(score=weighted_sum(weights, scores), weight=1)


class PMessage(ast.Message, PSharedMethods, PHasSimilarity):
    elements: List["PAlls"]

    def __repr__(self) -> str:
        return f"PMessage(fn={self.full_name} name={self.name}, elements={self.elements} d={self.depth})"

    def __init__(
        self,
        depth: int,
        parent: Optional[str],
        name: str,
        elements: List[ast.MessageElement | "PAlls"],
        parent_ast: Optional[ast.Message] = None,
    ):
        self.full_name = set_fullname(parent, name)
        self.depth = depth
        super().__init__(
            name, [ast_map(el, self.full_name, self.depth + 1, self) for el in elements if not isinstance(el, ast.Comment)]
        )
        self.parent_ast = parent_ast

    def nested(self) -> List["PNestedObjs"]:
        nested_elements: List[PNestedObjs] = []
        for el in self.elements:
            if isinstance(el, PNestedObjs):
                nested_elements.append(el)
                if isinstance(el, PMessage):
                    nested_elements.extend(el.nested())
        return nested_elements

    def _filter_by_type(self, element_type):
        return [el for el in self.elements if isinstance(el, element_type)]

    def flatten(self):
        def recurse(obj: PMessage):
            out = []
            for el in obj.elements:
                out.append(el)
                if isinstance(el, PMessage):
                    out.extend(recurse(el))
            return out

        return recurse(self)

    def fields(self) -> List[PField]:
        return self._filter_by_type((PField))

    def map_fields(self) -> List[PMapField]:
        return self._filter_by_type((PMapField))

    def oneofs(self) -> List["POneOf"]:
        return self._filter_by_type((POneOf))

    def messages(self) -> List["PMessage"]:
        return self._filter_by_type((PMessage))

    def enums(self) -> List["PEnum"]:
        return self._filter_by_type((PEnum))

    def __len__(self):
        return len(self.elements)

    def similarity_with(
        self,
        other: "PMessage",
        REGISTER1,
        REGISTER2,
        callstack: Optional[set] = None,
        _cache: Optional[dict] = None,
    ) -> SimilarityOutput:
        if not isinstance(other, PMessage):
            return SimilarityOutput.hard_penalization()

        if _cache is None:
            _cache = {}
        if callstack is None:
            callstack = set()
        key = (self.full_name, other.full_name)
        if key in _cache:
            return SimilarityOutput(
                weight=1,
                score=_cache[key],
            )
        if key in callstack:
            print(
                f"Recursive loop detected for {key}, returning {min(len(self), len(other))}"
            )
            return SimilarityOutput(
                weight=1,
                score=min(len(self), len(other)) / max(len(self), len(other)),
            )

        callstack.add(key)
        hg_path = HungarianPathGenerator(
            lambda x, y: x.similarity_with(
                y, REGISTER1, REGISTER2, callstack, _cache
            ).score
        ).compute(self.elements, other.elements)
        _cache[key] = hg_path.score
        callstack.remove(key)
        return SimilarityOutput(weight=1, score=hg_path.score)

    def get_fast_signature_counter(self) -> Counter:
        c = Counter()

        def get_key(el):
            if isinstance(el, PField):
                return (
                    f"primit:{el.type}" if el.is_primitive() else "customfield",
                    el.cardinality,
                )
            if isinstance(el, PMapField):
                return (
                    "primitmapfield" if el.is_primitive() else "custommapfield",
                    el.cardinality,
                )
            if isinstance(el, PEnum):
                return el.get_fast_signature_counter().most_common(1)[0]
            if isinstance(el, POneOf):
                return "oneof", None
            if isinstance(el, PMessage):
                return "message", None
            return None, None  # fallback for unknown types

        for el in self.elements:
            key, val = get_key(el)
            if key:
                c.update([(key, val)])

        return c


class POneOf(ast.OneOf, PSharedMethods, PHasSimilarity):
    def __init__(
        self,
        depth: int,
        parent: Optional[str],
        name: str,
        elements: List[ast.OneOfElement],
        parent_ast: Optional[ast.Message] = None,
    ):
        self.full_name = set_fullname(parent, name)
        self.depth = depth
        super().__init__(
            name,
            [ast_map(el, self.full_name, self.depth + 1, self) for el in elements if not isinstance(el, ast.Comment)],
        )
        self.parent_ast = parent_ast

    def __len__(self):
        return len(self.elements)

    def fields(self) -> List["PField"]:
        return [f for f in self.elements if isinstance(f, PField)]

    def similarity_with(
        self, other: "POneOf", REGISTER1, REGISTER2, callstack=None, _cache=None
    ) -> SimilarityOutput:
        if not isinstance(other, POneOf):
            return SimilarityOutput.hard_penalization()
        # Split into primitive fields only
        self_fields, other_fields = self.fields(), other.fields()
        hg_path = HungarianPathGenerator(
            lambda x, y: x.similarity_with(
                y, REGISTER1, REGISTER2, callstack, _cache
            ).score
        ).compute(self_fields, other_fields)
        score = ExactSimilarityDefinition.merge_hungarian_path_score(
            hg_path.score, self_fields, other_fields
        )
        return SimilarityOutput(weight=hg_path.weight, score=score)


class PEnum(ast.Enum, PSharedMethods, PHasSimilarity):
    def __repr__(self) -> str:
        return f"PEnum(fn={self.full_name} name={self.name}, len={len(self.elements)} d={self.depth})"

    def __init__(
        self,
        depth: int,
        parent: Optional[str],
        name: str,
        elements: List[ast.EnumElement],
        parent_ast: Optional[ast.Message] = None,
    ):
        self.full_name = set_fullname(parent, name)
        self.depth = depth
        super().__init__(
            name, [ast_map(el, self.full_name, self.depth + 1, self) for el in elements if not isinstance(el, ast.Comment)]
        )
        self.parent_ast = parent_ast

    def __len__(self):
        return len(self.elements)

    def similarity_with(self, other: "PEnum", *_, **__) -> SimilarityOutput:
        if not isinstance(other, PEnum):
            return SimilarityOutput.hard_penalization()
        return SimilarityOutput(
            weight=1,
            score=ExactSimilarityDefinition.hard_length_similarity(
                len(self), len(other)
            ),
        )

    def get_fast_signature_counter(self) -> Counter:
        return Counter([("enum", None)])


PBasicFields = PField | PMapField
PNestedObjs = PMessage | PEnum
PAlls = PField | PMapField | PMessage | POneOf | PEnum

####################################################################################################


def ast_map(
    x: ast.MessageElement,
    parent=None,
    depth=0,
    parent_ast: Optional[ast.Message] = None,
) -> PAlls:
    if isinstance(x, ast.Field):
        return PField(
            depth,
            parent,
            x.name,
            x.number,
            x.type,
            x.cardinality,
            x.options,
            parent_ast=parent_ast,
        )
    elif isinstance(x, ast.MapField):
        return PMapField(
            depth,
            parent,
            x.name,
            x.number,
            x.key_type,
            x.value_type,
            x.options,
            parent_ast=parent_ast,
        )
    elif isinstance(x, ast.Message):
        return PMessage(
            depth,
            parent,
            x.name,
            [ast_map(e, parent, depth + 1, x) for e in x.elements],
            parent_ast=parent_ast,
        )
    elif isinstance(x, ast.OneOf):
        return POneOf(
            depth,
            parent,
            x.name,
            [ast_map(e, parent, depth + 1, x) for e in x.elements],
            parent_ast=parent_ast,
        )
    elif isinstance(x, ast.Enum):
        return PEnum(
            depth,
            parent,
            x.name,
            [ast_map(e, parent, depth + 1, x) for e in x.elements],
            parent_ast=parent_ast,
        )
    else:
        return x


def get_struct_by_name(name: str, REGISTER) -> PMessage | PEnum | None:
    if name in REGISTER:
        return REGISTER[name]

    parts = name.split(".")
    for depth in range(1, len(parts) + 1):
        search_parts = ".".join(parts[-depth:])
        candidates = [key for key in REGISTER.keys() if key == search_parts]
        if len(candidates) == 1:
            key = candidates[0]
            return get_struct_by_name(key, REGISTER)

    struct = next((s for s in REGISTER.values() if s.name == name), None)
    if struct is not None:
        return struct
    raise ValueError(f"Proto {name} not found")


def get_pkg(file_elements: List[ast.FileElement]) -> ast.Package:
    return next(
        (el for el in file_elements if isinstance(el, ast.Package)),
        ast.Package(""),
    )


####################################################################################################
class PFile(ast.File):
    @staticmethod
    def from_file(file_path: Path) -> "PFile":
        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")
        parsed_ast = Parser().parse(file_path.read_text("utf-8"))
        return PFile.from_ast(parsed_ast)

    @staticmethod
    def from_ast(file: ast.File) -> "PFile":
        if not isinstance(file, ast.File):
            raise TypeError("Input must be an instance of ast.File")
        p_file_elements: List[PAlls | ast.MessageElement] = []
        pkg = get_pkg(file.file_elements)
        for el in file.file_elements:
            p_el = el
            if isinstance(el, (ast.Message, ast.Enum)):
                p_el = ast_map(el, parent=pkg.name)
            p_file_elements.append(p_el)
        
        return PFile(syntax=file.syntax, file_elements=p_file_elements)

    def __init__(self, syntax: str, file_elements: List[ast.FileElement]):
        file_elements = [el for el in file_elements if not isinstance(el, ast.Comment)]
        super().__init__(syntax, file_elements=file_elements)
        self.pkg = next(
            (el for el in self.file_elements if isinstance(el, ast.Package)),
            ast.Package("nopkg"),
        )

    def _filter_elements(self, element_type):
        return [el for el in self.file_elements if isinstance(el, element_type)]

    def messages(self):
        return self._filter_elements(PMessage)

    def enums(self):
        return self._filter_elements(PEnum)

    def flatten_objs(self):
        nesteds = []
        for el in self.messages():
            nesteds.append(el)
            nesteds.extend(el.nested())
        return sorted(nesteds + [el for el in self.enums()], key=lambda x: x.depth)

    def flatten_alls(self):
        all_objs = []
        for el in self.file_elements:
            if isinstance(el, PMessage):
                all_objs.append(el)
                all_objs.extend(el.flatten())
            elif isinstance(el, PEnum):
                all_objs.append(el)
            elif isinstance(el, POneOf):
                all_objs.append(el)
                all_objs.extend(el.elements)
            elif isinstance(el, PAlls): # Fallback for fields/others
                all_objs.append(el)
        return sorted(all_objs, key=lambda x: x.depth)

    def similarity_matrix_with(self, other: "PFile") -> np.ndarray:
        REF_REGISTER_MAP = {el.full_name: el for el in self.flatten_objs()}
        REGISTER_MAP = {el.full_name: el for el in other.flatten_objs()}
        ref_objs = self._filter_elements((PEnum, PMessage))
        other_objs = other._filter_elements((PEnum, PMessage))
        callstack = set()
        _cache = {}
        hg = HungarianPathGenerator(
            lambda c1, c2: c1.similarity_with(
                c2, REF_REGISTER_MAP, REGISTER_MAP, callstack, _cache
            ).score
        )
        sim_mat = hg.compute(ref_objs, other_objs).S
        return pd.DataFrame(
            sim_mat,
            index=[el.full_name for el in ref_objs],
            columns=[el.full_name for el in other_objs],
        )

    def graph(self):
        REGISTER_MAP = {el.full_name: el for el in self.flatten_objs()}
        import networkx as nx

        G = nx.DiGraph()
        for obj in self.messages():
            G.add_node(obj.full_name, ast=obj)
        for obj in self.enums():
            G.add_node(obj.full_name, ast=obj)

        def recursively_add(G, node: PMessage | PEnum):
            if isinstance(node, PEnum):
                return
            for el in node.elements:
                if isinstance(el, PField):
                    if not el.is_primitive():
                        ref = get_struct_by_name(el.type, REGISTER_MAP)
                        if ref is not None and ref.depth == 0 and node.depth == 0:
                            G.add_edge(node.full_name, ref.full_name)
                if isinstance(el, PMapField):
                    if not el.is_primitive():
                        key_ref = (
                            get_struct_by_name(el.key_type, REGISTER_MAP)
                            if el.key_type not in PROTOBUF_TYPES
                            else None
                        )
                        value_ref = (
                            get_struct_by_name(el.value_type, REGISTER_MAP)
                            if el.value_type not in PROTOBUF_TYPES
                            else None
                        )
                        if (
                            key_ref is not None
                            and key_ref.depth == 0
                            and node.depth == 0
                        ):
                            G.add_edge(node.full_name, key_ref.full_name)
                        if (
                            value_ref is not None
                            and value_ref.depth == 0
                            and node.depth == 0
                        ):
                            G.add_edge(node.full_name, value_ref.full_name)
                if isinstance(el, POneOf):
                    for oe in el.elements:
                        if isinstance(oe, PField):
                            if not oe.is_primitive():
                                ref = get_struct_by_name(oe.type, REGISTER_MAP)
                                if (
                                    ref is not None
                                    and ref.depth == 0
                                    and node.depth == 0
                                ):
                                    G.add_edge(node.full_name, ref.full_name)
                if isinstance(el, (PMessage, PEnum)):
                    recursively_add(G, el)

        for _, node in G.nodes(data=True):
            if isinstance(node["ast"], PMessage):
                recursively_add(G, node["ast"])
        return G