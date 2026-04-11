#!/usr/bin/env python3
"""Standalone smoke tests for the UniVL -> QFormer -> T5 caption decoder.

The script intentionally uses synthetic video features so it can isolate decoder
initialization, checkpoint loading, T5 loss, and generation from the dataset.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from types import SimpleNamespace
from typing import Dict, Tuple

import torch


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

LOGGER = logging.getLogger("test_t5_decoder")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test the T5 decoder path used by modules/modeling.py."
    )
    parser.add_argument("--mode", choices=["both", "raw_t5", "univl"], default="both")
    parser.add_argument("--t5_model", default="google/flan-t5-small")
    parser.add_argument("--bert_model", default="bert-base-uncased")
    parser.add_argument("--visual_model", default="visual-base")
    parser.add_argument("--cross_model", default="cross-base")
    parser.add_argument("--decoder_model", default="decoder-base")
    parser.add_argument("--init_model", default=None, help="Optional UniVL checkpoint path.")
    parser.add_argument("--cache_dir", default="")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--caption", default="a person is cooking in a kitchen")
    parser.add_argument("--prompt", default="A video of")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_words", type=int, default=20)
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--video_dim", type=int, default=768)
    parser.add_argument("--max_txt_len", type=int, default=24)
    parser.add_argument("--num_query_token", type=int, default=8)
    parser.add_argument("--qformer_vision_width", type=int, default=768)
    parser.add_argument("--qformer_checkpoint", default="")
    parser.add_argument("--qformer_checkpoint_file", default="")
    parser.add_argument("--qformer_checkpoint_local_files_only", action="store_true")
    parser.add_argument("--visual_num_hidden_layers", type=int, default=1)
    parser.add_argument("--cross_num_hidden_layers", type=int, default=1)
    parser.add_argument("--text_num_hidden_layers", type=int, default=1)
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Use cached HF files only.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available.")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tensor_stats(name: str, value: torch.Tensor) -> str:
    detached = value.detach()
    finite = bool(torch.isfinite(detached.float()).all().item())
    return (
        f"{name}: shape={tuple(detached.shape)} dtype={detached.dtype} "
        f"device={detached.device} finite={finite}"
    )


def load_checkpoint(path: str | None) -> Dict[str, torch.Tensor] | None:
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")

    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(state)!r}")

    interesting_prefixes = ("t5_model.", "t5_proj.", "Qformer.", "query_tokens")
    counts = {
        prefix: sum(1 for key in state.keys() if key.startswith(prefix))
        for prefix in interesting_prefixes
    }
    LOGGER.info("Loaded checkpoint: %s", path)
    LOGGER.info("Checkpoint decoder-related key counts: %s", counts)
    return state


def summarize_trainable_params(model: torch.nn.Module) -> None:
    total = 0
    trainable = 0
    by_prefix: Dict[str, Tuple[int, int]] = {}
    for name, param in model.named_parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
        prefix = name.split(".", 1)[0]
        old_total, old_trainable = by_prefix.get(prefix, (0, 0))
        by_prefix[prefix] = (
            old_total + count,
            old_trainable + (count if param.requires_grad else 0),
        )

    LOGGER.info("Parameters: total=%s trainable=%s", f"{total:,}", f"{trainable:,}")
    for prefix, (prefix_total, prefix_trainable) in sorted(by_prefix.items()):
        LOGGER.info(
            "  %-16s total=%12s trainable=%12s",
            prefix,
            f"{prefix_total:,}",
            f"{prefix_trainable:,}",
        )


def make_task_config(args: argparse.Namespace, device: torch.device) -> SimpleNamespace:
    return SimpleNamespace(
        do_pretrain=False,
        do_train=False,
        do_eval=True,
        stage_two=True,
        task_type="caption",
        datatype="synthetic",
        local_rank=0,
        world_size=1,
        n_gpu=1,
        batch_size=args.batch_size,
        batch_size_val=args.batch_size,
        n_pair=1,
        margin=0.1,
        hard_negative_rate=0.5,
        negative_weighting=1,
        use_mil=False,
        sampled_use_mil=False,
        max_words=args.max_words,
        max_frames=args.max_frames,
        video_dim=args.video_dim,
        text_num_hidden_layers=args.text_num_hidden_layers,
        visual_num_hidden_layers=args.visual_num_hidden_layers,
        cross_num_hidden_layers=args.cross_num_hidden_layers,
        decoder_num_hidden_layers=1,
        freeze_vit=False,
        scst=False,
        beam_size=args.beam_size,
        t5_model=args.t5_model,
        max_txt_len=args.max_txt_len,
        num_query_token=args.num_query_token,
        qformer_vision_width=args.qformer_vision_width,
        qformer_checkpoint=args.qformer_checkpoint,
        qformer_checkpoint_file=args.qformer_checkpoint_file,
        qformer_checkpoint_local_files_only=args.qformer_checkpoint_local_files_only,
        lora=args.lora,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        cache_dir=args.cache_dir,
        init_model=args.init_model,
        device=str(device),
    )


def make_synthetic_batch(model, args: argparse.Namespace, device: torch.device):
    batch_size = args.batch_size
    input_ids = torch.zeros(batch_size, 1, args.max_words, dtype=torch.long, device=device)
    segment_ids = torch.zeros_like(input_ids)
    input_mask = torch.zeros_like(input_ids)
    input_mask[:, :, :2] = 1
    if args.max_words > 1:
        input_ids[:, :, 1] = 102

    video = torch.randn(batch_size, 1, args.max_frames, args.video_dim, device=device)
    video_mask = torch.ones(batch_size, 1, args.max_frames, dtype=torch.long, device=device)

    input_caption_ids = torch.ones(batch_size, 1, args.max_words, dtype=torch.long, device=device)
    decoder_mask = torch.ones_like(input_caption_ids)

    t5_tokens = model.t5_tokenizer(
        [args.caption] * batch_size,
        padding="max_length",
        truncation=True,
        max_length=args.max_txt_len,
        return_tensors="pt",
    ).to(device)

    t5_label_ids = t5_tokens.input_ids

    # The real dataloader's pairs_output_caption_ids is BERT-tokenized. We do
    # not need the BERT tokenizer here; a valid-but-different ID stream is enough
    # to expose that the current forward path is not using T5 target IDs.
    max_fake_vocab_id = max(10, min(int(model.t5_model.config.vocab_size) - 1, 30521))
    fake_word_ids = []
    for index, token in enumerate(args.caption.split()):
        checksum = sum(ord(char) for char in token) + index * 997
        fake_word_ids.append(999 + (checksum % (max_fake_vocab_id - 999)))
    fake_bert_ids = [101] + fake_word_ids + [102]
    fake_bert_ids = fake_bert_ids[: args.max_words]

    bert_like_label_ids = torch.zeros(batch_size, args.max_words, dtype=torch.long, device=device)
    if fake_bert_ids:
        bert_like_label_ids[:, : len(fake_bert_ids)] = torch.tensor(
            fake_bert_ids, dtype=torch.long, device=device
        )

    # Mimic the legacy BERT-caption tensor shape consumed by forward().
    output_caption_ids = bert_like_label_ids.view(batch_size, 1, -1)
    t5_output_caption_ids = t5_label_ids.view(batch_size, 1, -1)

    return {
        "input_ids": input_ids,
        "segment_ids": segment_ids,
        "input_mask": input_mask,
        "video": video,
        "video_mask": video_mask,
        "input_caption_ids": input_caption_ids,
        "decoder_mask": decoder_mask,
        "output_caption_ids": output_caption_ids,
        "t5_output_caption_ids": t5_output_caption_ids,
        "t5_label_ids": t5_label_ids,
        "bert_like_label_ids": bert_like_label_ids,
    }


def assert_finite_scalar(name: str, value: torch.Tensor) -> None:
    if value.ndim != 0:
        raise AssertionError(f"{name} should be scalar, got shape={tuple(value.shape)}")
    if not torch.isfinite(value.detach().float()).item():
        raise AssertionError(f"{name} is not finite: {value}")


def run_raw_t5(args: argparse.Namespace, device: torch.device) -> None:
    from transformers import T5ForConditionalGeneration, T5TokenizerFast

    LOGGER.info("=== Raw T5 sanity test ===")
    tokenizer = T5TokenizerFast.from_pretrained(args.t5_model)
    model = T5ForConditionalGeneration.from_pretrained(args.t5_model).to(device)
    model.eval()

    inputs = tokenizer([args.prompt], return_tensors="pt").to(device)
    labels = tokenizer([args.caption], return_tensors="pt").input_ids.to(device)
    labels = labels.masked_fill(labels == tokenizer.pad_token_id, -100)

    with torch.no_grad():
        outputs = model(**inputs, labels=labels, return_dict=True)
        generated = model.generate(**inputs, num_beams=args.beam_size, max_length=args.max_txt_len)

    assert_finite_scalar("raw_t5_loss", outputs.loss)
    LOGGER.info("raw_t5_loss=%.6f", float(outputs.loss.detach().float().cpu()))
    LOGGER.info("raw_t5_generated=%r", tokenizer.batch_decode(generated, skip_special_tokens=True))


def run_univl(args: argparse.Namespace, device: torch.device) -> None:
    from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
    from modules.modeling import UniVL

    LOGGER.info("=== UniVL T5 decoder integration test ===")
    task_config = make_task_config(args, device)
    cache_dir = args.cache_dir or os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), "distributed")
    state_dict = load_checkpoint(args.init_model)

    model = UniVL.from_pretrained(
        args.bert_model,
        args.visual_model,
        args.cross_model,
        args.decoder_model,
        cache_dir=cache_dir,
        state_dict=state_dict,
        task_config=task_config,
    )
    model.prompt = " " + args.prompt.strip()
    model.to(device)
    model.eval()
    summarize_trainable_params(model)

    batch = make_synthetic_batch(model, args, device)
    with torch.no_grad():
        forward_loss, visual_output = model(
            batch["input_ids"],
            batch["segment_ids"],
            batch["input_mask"],
            batch["video"],
            batch["video_mask"],
            input_caption_ids=batch["input_caption_ids"],
            decoder_mask=batch["decoder_mask"],
            output_caption_ids=batch["output_caption_ids"],
            t5_output_caption_ids=batch["t5_output_caption_ids"],
        )
        assert forward_loss is not None
        assert_finite_scalar("forward_loss_current_path", forward_loss)
        LOGGER.info("forward_loss_current_path=%.6f", float(forward_loss.detach().float().cpu()))
        LOGGER.info(tensor_stats("visual_output", visual_output))

        flat_video_mask = batch["video_mask"].view(-1, batch["video_mask"].shape[-1])
        inputs_embeds, encoder_atts = model._build_t5_encoder_inputs(visual_output, flat_video_mask)
        LOGGER.info(tensor_stats("t5_inputs_embeds", inputs_embeds))
        LOGGER.info(tensor_stats("t5_encoder_attention", encoder_atts))

        current_label_loss = model._get_t5_caption_loss(
            visual_output,
            flat_video_mask,
            batch["bert_like_label_ids"],
        )
        t5_label_loss = model._get_t5_caption_loss(
            visual_output,
            flat_video_mask,
            batch["t5_label_ids"],
        )
        assert_finite_scalar("current_label_loss", current_label_loss)
        assert_finite_scalar("t5_label_loss", t5_label_loss)
        LOGGER.info("current_label_loss=%.6f", float(current_label_loss.detach().float().cpu()))
        LOGGER.info("t5_label_loss=%.6f", float(t5_label_loss.detach().float().cpu()))
        LOGGER.info(
            "Loss note: forward() currently consumes output_caption_ids; "
            "for real dataloader batches that tensor is BERT-tokenized, while "
            "pairs_t5_output_caption_ids is the T5-tokenized target."
        )

        generated_ids = model.generate_caption_ids(
            visual_output,
            flat_video_mask,
            num_beams=args.beam_size,
            max_length=args.max_txt_len,
        )
        LOGGER.info(tensor_stats("generated_ids", generated_ids))
        LOGGER.info(
            "generated_text=%r",
            model.t5_tokenizer.batch_decode(generated_ids, skip_special_tokens=True),
        )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    if args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    device = choose_device(args.device)
    LOGGER.info("Using device=%s", device)

    if args.mode in ("both", "raw_t5"):
        run_raw_t5(args, device)
    if args.mode in ("both", "univl"):
        run_univl(args, device)
    LOGGER.info("All requested T5 decoder checks passed.")


if __name__ == "__main__":
    main()
