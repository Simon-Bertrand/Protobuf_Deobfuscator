
import argparse
import sys
import logging
from protobuf_deobfuscator.pipeline import approximate_protobuf_qap
from proto_schema_parser.parser import Parser
from pathlib import Path
def main():
    parser = argparse.ArgumentParser(
        description="Protobuf Deobfuscator - A QAP-based heuristic solver for restoring obfuscated Protobuf schemas."
    )

    parser.add_argument(
        "obfuscated_proto",
        help="Path to the obfuscated .proto file (Target)",
        type=Path
    )
    
    parser.add_argument(
        "reference_proto",
        help="Path to the reference .proto file (Source/Ground Truth)",
        type=Path
    )

    # Tuning Parameters
    parser.add_argument(
        "--top-k", "-k",
        help="Number of candidates to select for expensive Cross-Ref scoring (Default: 16)",
        type=int,
        default=16
    )

    parser.add_argument(
        "--threshold", "-t",
        help="Minimum similarity score [0.0-1.0] required to accept a match (Default: 0.2)",
        type=float,
        default=0.2
    )

    parser.add_argument(
        "--verbose", "-v",
        help="Enable verbose logging",
        action="store_true"
    )

    parser.add_argument(
        "--format", "-f",
        help="Output format: text (default) or json",
        type=str,
        default="text",
        choices=["text", "json"]
    )

    args = parser.parse_args()

    # Configure Logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    # Use basicConfig to ensure a handler is set up if none exists
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    logging.getLogger().setLevel(log_level)

    mapping = approximate_protobuf_qap(
        obfuscated_proto_path=args.obfuscated_proto,
        ref_proto_path=args.reference_proto,
        min_score_threshold=args.threshold,
        TOP_K=args.top_k # Need to ensure pipeline/bottom_up accepts this
    )

    if args.format == "json":
        import json
        output = {str(k): v for k, v in mapping.items()}
        print(json.dumps(output, indent=2))
    else:
        print(f"Found {len(mapping)} mappings:")
        for (ref, obf), score in mapping.items():
            print(f"{ref} -> {obf} (Score: {score:.3f})")

if __name__ == "__main__":
    main()
