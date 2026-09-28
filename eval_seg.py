#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate XMask instance segmentation with Mean IoU C."""
import argparse

from xmask.evaluation import (
    SEG_ENERGY_THRESHOLD,
    SEG_GAUSSIAN_KERNEL_SIZE,
    SEG_GAUSSIAN_SIGMA,
    SEG_GEOMETRY_WEIGHT,
    add_common_arguments,
    evaluate,
    prepare_evaluation,
    print_seg_result,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate XMask instance segmentation Mean IoU C."
    )
    add_common_arguments(parser)
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
        tasks=("seg",),
        print_freq=args.print_freq,
    )
    print_seg_result(results["seg"])


if __name__ == "__main__":
    main()
