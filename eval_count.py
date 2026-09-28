#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate XMask crowd-counting MAE and RMSE."""
import argparse

from xmask.evaluation import (
    add_common_arguments,
    evaluate,
    prepare_evaluation,
    print_count_result,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate XMask counting metrics.")
    add_common_arguments(parser)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="density-logit threshold used to count predictions (default: 0.0)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    loader, model, device = prepare_evaluation(args)
    results = evaluate(
        loader,
        model,
        device,
        tasks=("count",),
        count_threshold=args.threshold,
        print_freq=args.print_freq,
    )
    print_count_result(results["count"])


if __name__ == "__main__":
    main()
