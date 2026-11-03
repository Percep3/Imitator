"""Usage: python -m scripts.wsp_imitator_e1.run --help (development only)."""
import argparse
from pathlib import Path
from scripts.wsp_imitator_e1.harness import Study, repo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    repo.shared_parser(parser)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/wsp_imitator_e1"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--role", choices=("reproduction", "discovery", "confirmation"), default="reproduction")
    parser.add_argument("--condition", default="clean")
    parser.add_argument("--patch", type=Path)
    parser.add_argument("--probe-layer", action="append")
    args = parser.parse_args()
    study = Study(args)
    if args.patch:
        study.weights.load_patch(args.patch, study.checkpoint_hash)
    if args.probe_layer:
        if args.patch:
            parser.error("probes start from original; use activation_intervention for causal diagnostics")
        study.teacher_forced_probe(args.probe_layer)
    elif args.patch:
        study.evaluate_paired(args.role, args.condition)
    else:
        study.evaluate_original(args.role, args.condition)


if __name__ == "__main__":
    main()
