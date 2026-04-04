import argparse
import os
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataloaders.dataloader_msrvtt_caption_t5 import MSRVTT_Caption_T5_DataLoader
from inference.caption_generator_t5 import eval_epoch
from modules.modeling_t5 import UniVL_T5
from modules.tokenization import BertTokenizer
from transformers import T5Tokenizer


def parse_args():
    parser = argparse.ArgumentParser("Smoke test two-path MSRVTT caption pipeline")
    parser.add_argument(
        "--features_path",
        type=str,
        default="/Users/mac/Documents/NCKH/Source/UniVL-Video-captioning/msrvtt_clip_vitl14_features.pickle",
        help="Path to MSRVTT feature pickle",
    )
    parser.add_argument("--data_path", type=str, default="data/msrvtt/MSRVTT_data.json")
    parser.add_argument("--train_csv", type=str, default="data/msrvtt/MSRVTT_train.7k.csv")
    parser.add_argument("--val_csv", type=str, default="data/msrvtt/MSRVTT_JSFUSION_test.csv")
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased")
    parser.add_argument("--t5_model", type=str, default="t5-base")
    parser.add_argument("--max_words", type=int, default=24)
    parser.add_argument("--max_frames", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_subset", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--caption_prompt_mode", type=str, default="fixed", choices=["empty", "fixed"])
    parser.add_argument("--caption_prompt_text", type=str, default="What does the video describe?")
    parser.add_argument("--output_dir", type=str, default="output/smoke_two_path")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def bert_decode(ids, tokenizer):
    tokens = tokenizer.convert_ids_to_tokens(ids)
    return " ".join([t for t in tokens if t not in ["[PAD]"]])


class PrintLogger:
    def info(self, msg, *args):
        print((msg % args) if args else msg)

    def warning(self, msg, *args):
        print((msg % args) if args else msg)


class FirstNSampler(Sampler):
    def __init__(self, n):
        self.n = int(n)

    def __iter__(self):
        return iter(range(self.n))

    def __len__(self):
        return self.n


def build_task_config(feature_dim, args):
    return SimpleNamespace(
        max_words=args.max_words,
        max_frames=args.max_frames,
        stage_two=True,
        train_sim_after_cross=False,
        text_num_hidden_layers=2,
        visual_num_hidden_layers=2,
        cross_num_hidden_layers=2,
        do_pretrain=False,
        task_type="caption",
        itc_proj_dim=128,
        itc_init_temp=0.07,
        itm_use_richer_fusion=True,
        use_itc_loss=True,
        use_itm_loss=True,
        itc_gather_distributed=False,
        current_itc_weight=0.05,
        current_itm_weight=0.05,
        itc_weight=0.05,
        itm_weight=0.05,
        batch_size=args.batch_size,
        n_gpu=1,
        use_mil=False,
        n_pair=1,
        margin=0.1,
        negative_weighting=1,
        hard_negative_rate=0.5,
        local_rank=0,
        video_dim=feature_dim,
        t5_model=args.t5_model,
        use_qformer_aux_loss=False,
        datatype="msrvtt",
        output_dir=args.output_dir,
    )


def main():
    args = parse_args()
    set_seed(args.seed)

    if not os.path.exists(args.features_path):
        raise FileNotFoundError("Feature pickle not found: {}".format(args.features_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = PrintLogger()

    bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
    t5_tokenizer = T5Tokenizer.from_pretrained(args.t5_model)

    train_dataset = MSRVTT_Caption_T5_DataLoader(
        csv_path=args.train_csv,
        json_path=args.data_path,
        features_path=args.features_path,
        bert_tokenizer=bert_tokenizer,
        t5_tokenizer=t5_tokenizer,
        max_words=args.max_words,
        max_frames=args.max_frames,
        split_type="train",
        caption_prompt_mode=args.caption_prompt_mode,
        caption_prompt_text=args.caption_prompt_text,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    first_batch = next(iter(train_loader))
    (
        input_ids,
        input_mask,
        segment_ids,
        video,
        video_mask,
        pairs_masked_text,
        pairs_token_labels,
        masked_video,
        video_labels_index,
        pairs_input_caption_ids,
        pairs_decoder_mask,
        pairs_output_caption_ids,
        align_input_ids,
        align_mask,
        align_segment,
        sample_video_ids,
    ) = first_batch

    # Decode checks to verify prompt/alignment separation + teacher forcing.
    prompt_ids = [int(x) for x in input_ids[0, 0].tolist() if int(x) > 0]
    align_ids = [int(x) for x in align_input_ids[0, 0].tolist() if int(x) > 0]
    out_ids = [int(x) for x in pairs_output_caption_ids[0, 0].tolist() if int(x) >= 0]
    in_ids = [int(x) for x in pairs_input_caption_ids[0, 0].tolist()[: len(out_ids)]]

    print("PROMPT_DECODED:", bert_decode(prompt_ids, bert_tokenizer))
    print("ALIGN_DECODED:", bert_decode(align_ids, bert_tokenizer))
    print("T5_TARGET_DECODED:", t5_tokenizer.decode(out_ids, skip_special_tokens=True))

    assert int(pairs_input_caption_ids[0, 0, 0].item()) == int(t5_tokenizer.pad_token_id)
    assert in_ids[1:] == out_ids[:-1], "Teacher forcing shift mismatch"

    feature_dim = int(video.shape[-1])
    task_config = build_task_config(feature_dim, args)
    model = UniVL_T5.from_pretrained(
        args.bert_model,
        "visual-base",
        "cross-base",
        "decoder-base",
        state_dict=None,
        task_config=task_config,
    ).to(device)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    train_batch = tuple(t.to(device) for t in first_batch)
    (
        input_ids,
        input_mask,
        segment_ids,
        video,
        video_mask,
        pairs_masked_text,
        pairs_token_labels,
        masked_video,
        video_labels_index,
        pairs_input_caption_ids,
        pairs_decoder_mask,
        pairs_output_caption_ids,
        align_input_ids,
        align_mask,
        align_segment,
        sample_video_ids,
    ) = train_batch

    output = model(
        input_ids,
        segment_ids,
        input_mask,
        video,
        video_mask,
        pairs_masked_text=pairs_masked_text,
        pairs_token_labels=pairs_token_labels,
        masked_video=masked_video,
        video_labels_index=video_labels_index,
        input_caption_ids=pairs_input_caption_ids,
        decoder_mask=pairs_decoder_mask,
        output_caption_ids=pairs_output_caption_ids,
        align_input_ids=align_input_ids,
        align_mask=align_mask,
        align_segment=align_segment,
        sample_video_ids=sample_video_ids,
    )

    assert isinstance(output, dict) and "loss" in output and "decoder_loss" in output
    assert torch.isfinite(output["loss"]).item(), "Total loss is not finite"
    assert torch.isfinite(output["decoder_loss"]).item(), "Decoder loss is not finite"

    optimizer.zero_grad()
    output["loss"].backward()
    optimizer.step()
    print("TRAIN_FORWARD_BACKWARD_OK")

    # Inference path should work without alignment tensors.
    model.eval()
    with torch.no_grad():
        dec_loss = model(
            input_ids,
            segment_ids,
            input_mask,
            video,
            video_mask,
            input_caption_ids=pairs_input_caption_ids,
            decoder_mask=pairs_decoder_mask,
            output_caption_ids=pairs_output_caption_ids,
            align_input_ids=None,
            align_mask=None,
            align_segment=None,
        )
    assert dec_loss is not None and torch.isfinite(dec_loss).item(), "Eval loss failed without align tensors"
    print("INFERENCE_WITHOUT_ALIGN_OK")

    # Eval smoke with real feature pickle + beam search + hyp/ref outputs.
    val_dataset = MSRVTT_Caption_T5_DataLoader(
        csv_path=args.val_csv,
        json_path=args.data_path,
        features_path=args.features_path,
        bert_tokenizer=bert_tokenizer,
        t5_tokenizer=t5_tokenizer,
        max_words=args.max_words,
        max_frames=args.max_frames,
        split_type="test",
        caption_prompt_mode=args.caption_prompt_mode,
        caption_prompt_text=args.caption_prompt_text,
    )
    subset_size = min(args.eval_subset, len(val_dataset))
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=FirstNSampler(subset_size),
        num_workers=0,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    bleu4, avg_val_loss = eval_epoch(
        task_config,
        model,
        val_loader,
        t5_tokenizer,
        device,
        1,
        logger,
        nlgEvalObj=None,
        test_set=None,
    )

    hyp_path = os.path.join(args.output_dir, "hyp.txt")
    ref_path = os.path.join(args.output_dir, "ref.txt")
    assert os.path.exists(hyp_path), "hyp.txt was not generated"
    assert os.path.exists(ref_path), "ref.txt was not generated"
    assert os.path.getsize(hyp_path) > 0, "hyp.txt is empty"

    print("EVAL_BEAM_OK, BLEU4=%.4f, AVG_VAL_LOSS=%.6f" % (float(bleu4), float(avg_val_loss)))
    print("HYP_PATH:", hyp_path)
    print("REF_PATH:", ref_path)
    print("ALL_SMOKE_TESTS_PASSED")


if __name__ == "__main__":
    main()