#!/usr/bin/env python3
"""Print the visual_config.hidden_size passed into Blip2Base.init_Qformer."""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.module_visual import VisualConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show visual_config.hidden_size used by UniVL QFormer init."
    )
    parser.add_argument("--visual_model", default="visual-base")
    parser.add_argument("--cache_dir", default="")
    parser.add_argument("--video_dim", type=int, default=1024)
    parser.add_argument("--num_query_token", type=int, default=32)
    parser.add_argument("--qformer_vision_width", type=int, default=768)
    parser.add_argument("--local_rank", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = args.cache_dir or os.path.join(
        str(PYTORCH_PRETRAINED_BERT_CACHE), "distributed"
    )
    task_config = SimpleNamespace(
        local_rank=args.local_rank,
        video_dim=args.video_dim,
        num_query_token=args.num_query_token,
    )

    visual_config, _ = VisualConfig.get_config(
        args.visual_model,
        cache_dir=cache_dir,
        type_vocab_size=2,
        state_dict=None,
        task_config=task_config,
    )
    visual_config.vocab_size = task_config.video_dim

    print("UniVL QFormer init call:")
    print("  self.Qformer, self.query_tokens = Blip2Base.init_Qformer(")
    print(f"      num_query_token={args.num_query_token},")
    print(f"      vision_width={args.qformer_vision_width}")
    print("  )")
    print()
    print(f"visual_model: {args.visual_model}")
    print(f"visual_config.hidden_size: {visual_config.hidden_size}")
    print(f"visual_config.vocab_size/video_dim: {visual_config.vocab_size}")
    if args.qformer_vision_width != visual_config.hidden_size:
        print(
            "qformer_visual_proj: "
            f"Linear({visual_config.hidden_size}, {args.qformer_vision_width})"
        )
    else:
        print("qformer_visual_proj: Identity()")


if __name__ == "__main__":
    main()
