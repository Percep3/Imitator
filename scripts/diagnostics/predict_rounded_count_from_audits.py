"""Fase 0: predict pred_rescaled_to_rounded_count results from existing LOSO audits.

Reads the hist_quantity_minus_target_len histograms already stored in
fold*_test_eval.json (no GPU, no model load) and estimates the count_match_rate
the rounded-count mode would achieve, with and without a global additive bias.
Run before spending GPU time on re-evaluation; see
reports/report_2026-07-03_fase0_rounded_count_prediction.md for the decision it fed.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics


def load_histograms(pattern: str) -> dict[int, dict[float, int]]:
    folds = {}
    for path in sorted(glob.glob(pattern)):
        data = json.load(open(path))
        hist = data["mode_cuts"]["pred_raw"]["hist_quantity_minus_target_len"]
        folds[data["heldout_signer"]] = {float(k): v for k, v in hist.items()}
    return folds


def p_round_correct(hist: dict[float, int], bias: float) -> float:
    # bin label = quantity_delta rounded to 1 decimal, so |delta+bias|<0.5 is exact
    # up to the 0.05 bin-edge ambiguity (negligible for a go/no-go estimate)
    n = sum(hist.values())
    return sum(v for k, v in hist.items() if abs(k + bias) < 0.5) / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audits",
        default="/shared/Code/Sign-AI/outputs/loso_clean/fold*_test_eval.json",
    )
    parser.add_argument(
        "--biases", type=float, nargs="+", default=[0.0, 0.2, 0.3, 0.4, 0.5, 0.6]
    )
    args = parser.parse_args()

    folds = load_histograms(args.audits)
    if not folds:
        raise SystemExit(f"no audits matched {args.audits}")

    print("P(round(quantity+b) == target_len) per fold:")
    print(f"{'b':>5} " + " ".join(f"f{k:>5}" for k in folds) + "  media")
    for b in args.biases:
        ps = [p_round_correct(h, b) for h in folds.values()]
        print(
            f"{b:>5.1f} "
            + " ".join(f"{p:>6.3f}" for p in ps)
            + f"  {statistics.mean(ps):.3f}"
        )

    print("\nper-fold oracle bias ceiling (reference only, peeks at held-out):")
    for k, h in folds.items():
        best_p, best_b = max((p_round_correct(h, b / 100), b / 100) for b in range(0, 101, 5))
        print(f"  fold {k}: b*={best_b:.2f} -> P={best_p:.3f}")

    print("\nmean quantity bias per fold (negative = CIF undercounts):")
    for k, h in folds.items():
        n = sum(h.values())
        print(f"  fold {k}: {sum(kk * v for kk, v in h.items()) / n:+.3f}")


if __name__ == "__main__":
    main()
