#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import argparse
import sys

import torch

from deepspeed.ops.transformer.inference.config import DeepSpeedInferenceConfig
from deepspeed.ops.transformer.inference.op_binding.workspace import WorkspaceOp


def _print_env(device):
    print(f"torch.version: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    try:
        print(f"torch.cuda.driver_version: {torch.cuda.driver_version()}")
    except Exception as exc:
        print(f"torch.cuda.driver_version: unavailable ({exc})")
    props = torch.cuda.get_device_properties(device)
    print(f"device: {device} ({props.name})")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Repro script for CUDA error handling in DeepSpeed inference workspace allocation."
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-length", type=int, default=1)
    parser.add_argument("--max-out-tokens", type=int, default=1024)
    parser.add_argument("--min-out-tokens", type=int, default=1)
    return parser.parse_args()


def main():
    args = _parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available. This script requires a CUDA-capable setup.")
        return 1

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    _print_env(device)

    # Force CUDA initialization before loading the inference op.
    torch.empty(1, device=device)

    config = DeepSpeedInferenceConfig(
        hidden_size=args.hidden_size,
        intermediate_size=4 * args.hidden_size,
        heads=args.heads,
        num_hidden_layers=args.layers,
        dtype=torch.float16,
        mp_size=1,
        max_out_tokens=args.max_out_tokens,
        min_out_tokens=args.min_out_tokens,
    )

    workspace = WorkspaceOp(config)
    if "fallback" in workspace.allocate_workspace_func.__name__:
        raise RuntimeError(
            "DeepSpeed inference op did not load; CUDA extension is required to reproduce this issue."
        )

    print("Allocating inference workspace...")
    workspace.allocate_workspace(
        config.hidden_size,
        config.heads,
        args.prompt_length,
        args.batch_size,
        args.layers,
        config.mp_size,
        False,
        0,
        args.max_out_tokens,
        args.min_out_tokens,
    )
    print("Workspace allocation succeeded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
