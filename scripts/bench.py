from settings import initialize
initialize()

from src.mslm.benchmark.BLEU import main

def run(version: str, checkpoint: str, epoch: int, use_cached_results: bool = False):
    main(version, checkpoint, epoch, use_cached_results)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a model using BLEU score.")
    parser.add_argument("--version", type=str, required=True, help="Model version to evaluate.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint name to evaluate.")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch number of the checkpoint to evaluate.")
    parser.add_argument("--use-cached-results", action='store_true', help="Use cached BLEU results if available.")

    args = parser.parse_args()

    run(args.version, args.checkpoint, args.epoch, args.use_cached_results)

