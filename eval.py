#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run complete XMask counting and instance-segmentation evaluation."""
import argparse

from xmask.evaluation import (
    SEG_ENERGY_THRESHOLD,
    SEG_GAUSSIAN_KERNEL_SIZE,
    SEG_GAUSSIAN_SIGMA,
    SEG_GEOMETRY_WEIGHT,
    add_common_arguments,
    evaluate,
    prepare_evaluation,
    print_count_result,
    print_seg_result,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate XMask counting and instance segmentation."
    )
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
    print(
        "Segmentation parameters: "
        f"energy_threshold={SEG_ENERGY_THRESHOLD} "
        f"geometry_weight={SEG_GEOMETRY_WEIGHT} "
        f"gaussian=({SEG_GAUSSIAN_KERNEL_SIZE}, {SEG_GAUSSIAN_SIGMA:g})"
    )
    results = evaluate(
        loader,
        model,
        device,
        tasks=("count", "seg"),
        count_threshold=args.threshold,
        print_freq=args.print_freq,
    )
    print_count_result(results["count"])
    print_seg_result(results["seg"])


if __name__ == "__main__":
    main()
