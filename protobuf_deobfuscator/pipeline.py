
from protobuf_deobfuscator.ast import PFile
from protobuf_deobfuscator.graphs import ProtoGraphs

import logging
from protobuf_deobfuscator.biasing import generate_bias
from protobuf_deobfuscator.bottom_up import bottom_up_mapping



def approximate_protobuf_qap(obfuscated_proto_path, ref_proto_path, additional_bias = {}, min_score_threshold = 0.20, TOP_K=16):
    ref_protofile = PFile.from_file(ref_proto_path)
    obfuscated_proto = PFile.from_file(
       obfuscated_proto_path
    )
    graphs = ProtoGraphs(ref_protofile, obfuscated_proto)
    bias = generate_bias(graphs) | additional_bias

    callstack = set()
    mapping = bottom_up_mapping(
        graphs,
        log=logging.getLogger(),
        min_score_threshold=min_score_threshold,
        bias=bias,
        callstack=callstack,
        TOP_K=TOP_K
    )
    return mapping