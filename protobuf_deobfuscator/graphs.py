from functools import lru_cache

import networkx as nx

from protobuf_deobfuscator.ast import PFile


class ProtoGraphs:
    Gref: nx.DiGraph
    G: nx.DiGraph

    def __init__(self, ref_proto: PFile, proto: PFile):
        self.ref_proto = ref_proto
        self.proto = proto
        self.Gref = ref_proto.graph()
        self.G = proto.graph()
        self.REF_REGISTER_MAP = {el.full_name: el for el in ref_proto.flatten_objs()}
        self.REGISTER_MAP = {el.full_name: el for el in proto.flatten_objs()}
        self.ref_nodes = list(self.Gref.nodes)
        self.nodes = list(self.G.nodes)
        self.updated = False