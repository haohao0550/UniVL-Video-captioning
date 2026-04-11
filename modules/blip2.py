"""
 Copyright (c) 2023, anonymous.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import contextlib
import json
import logging
import os
import time
import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

# import lavis.common.dist_utils as dist_utils
# from lavis.common.dist_utils import download_cached_file
# from lavis.common.utils import is_url
# from lavis.common.logger import MetricLogger
from modules.base_model import BaseModel
from modules.Qformer import BertConfig, BertLMHeadModel
# from lavis.models.eva_vit import create_eva_vit_g
# from lavis.models.clip_vit import create_clip_vit_L
from transformers import BertTokenizer

logger = logging.getLogger(__name__)


class Blip2Base(BaseModel):
    @classmethod
    def init_tokenizer(cls, truncation_side="right"):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def init_Qformer(
            cls,
            num_query_token,
            vision_width,
            cross_attention_freq=2,
            qformer_checkpoint=None,
            qformer_checkpoint_file=None,
            local_files_only=False,
    ):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel.from_pretrained(
            "bert-base-uncased", config=encoder_config
        )
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        cls.load_qformer_checkpoint(
            Qformer,
            query_tokens,
            qformer_checkpoint,
            checkpoint_file=qformer_checkpoint_file,
            local_files_only=local_files_only,
        )
        return Qformer, query_tokens

    @staticmethod
    def load_qformer_checkpoint(
            qformer,
            query_tokens,
            checkpoint,
            checkpoint_file=None,
            local_files_only=False,
    ):
        if not checkpoint:
            return None

        target_state = qformer.state_dict()
        qformer_state = {}
        copied_query_tokens = False
        skipped = []

        for source_key, tensor in Blip2Base.iter_qformer_checkpoint_tensors(
                checkpoint,
                checkpoint_file=checkpoint_file,
                local_files_only=local_files_only,
        ):
            mapped_key = Blip2Base.map_blip2_qformer_key(source_key)
            if mapped_key is None:
                continue

            if mapped_key == "query_tokens":
                if tuple(tensor.shape) == tuple(query_tokens.shape):
                    query_tokens.data.copy_(tensor.to(dtype=query_tokens.dtype))
                    copied_query_tokens = True
                else:
                    skipped.append((source_key, mapped_key, tuple(tensor.shape), tuple(query_tokens.shape)))
                continue

            if mapped_key in target_state and tuple(tensor.shape) == tuple(target_state[mapped_key].shape):
                qformer_state[mapped_key] = tensor.to(dtype=target_state[mapped_key].dtype)
            elif mapped_key in target_state:
                skipped.append((source_key, mapped_key, tuple(tensor.shape), tuple(target_state[mapped_key].shape)))

        msg = qformer.load_state_dict(qformer_state, strict=False)
        logger.info(
            "Loaded QFormer checkpoint from %s: tensors=%d query_tokens=%s missing=%d unexpected=%d skipped_shape=%d",
            checkpoint,
            len(qformer_state),
            copied_query_tokens,
            len(msg.missing_keys),
            len(msg.unexpected_keys),
            len(skipped),
        )
        if skipped:
            logger.info("First skipped QFormer checkpoint keys: %s", skipped[:10])
        return msg

    @staticmethod
    def map_blip2_qformer_key(key):
        key = key.replace("module.", "")

        query_token_keys = {
            "query_tokens",
            "blip2.query_tokens",
            "model.query_tokens",
        }
        if key in query_token_keys or key.endswith(".query_tokens"):
            return "query_tokens"

        prefixes = (
            "qformer.",
            "Qformer.",
            "blip2.qformer.",
            "model.qformer.",
            "blip2opt.qformer.",
        )
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        else:
            return None

        if key.startswith(("bert.", "cls.")):
            return key
        return "bert." + key

    @staticmethod
    def iter_qformer_checkpoint_tensors(checkpoint, checkpoint_file=None, local_files_only=False):
        if os.path.isdir(checkpoint):
            yield from Blip2Base.iter_local_qformer_checkpoint_tensors(checkpoint, checkpoint_file)
            return
        if os.path.isfile(checkpoint):
            yield from Blip2Base.iter_checkpoint_file_tensors(checkpoint)
            return
        yield from Blip2Base.iter_hf_qformer_checkpoint_tensors(
            checkpoint,
            checkpoint_file=checkpoint_file,
            local_files_only=local_files_only,
        )

    @staticmethod
    def iter_local_qformer_checkpoint_tensors(checkpoint_dir, checkpoint_file=None):
        if checkpoint_file:
            yield from Blip2Base.iter_checkpoint_file_tensors(os.path.join(checkpoint_dir, checkpoint_file))
            return

        for filename in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            index_path = os.path.join(checkpoint_dir, filename)
            if os.path.exists(index_path):
                with open(index_path, "r", encoding="utf-8") as reader:
                    index = json.load(reader)
                qformer_files = sorted({
                    shard for key, shard in index.get("weight_map", {}).items()
                    if key == "query_tokens" or ".query_tokens" in key or "qformer." in key.lower()
                })
                for shard in qformer_files:
                    yield from Blip2Base.iter_checkpoint_file_tensors(os.path.join(checkpoint_dir, shard))
                return

        for filename in ("model.safetensors", "pytorch_model.bin"):
            path = os.path.join(checkpoint_dir, filename)
            if os.path.exists(path):
                yield from Blip2Base.iter_checkpoint_file_tensors(path)
                return

        raise FileNotFoundError("No supported checkpoint file found in {}".format(checkpoint_dir))

    @staticmethod
    def iter_hf_qformer_checkpoint_tensors(repo_id, checkpoint_file=None, local_files_only=False):
        from huggingface_hub import hf_hub_download

        if checkpoint_file:
            path = hf_hub_download(repo_id, checkpoint_file, local_files_only=local_files_only)
            yield from Blip2Base.iter_checkpoint_file_tensors(path)
            return

        index_filenames = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
        last_error = None
        for index_filename in index_filenames:
            try:
                index_path = hf_hub_download(repo_id, index_filename, local_files_only=local_files_only)
            except Exception as exc:
                last_error = exc
                continue

            with open(index_path, "r", encoding="utf-8") as reader:
                index = json.load(reader)
            qformer_files = sorted({
                shard for key, shard in index.get("weight_map", {}).items()
                if key == "query_tokens" or ".query_tokens" in key or "qformer." in key.lower()
            })
            if not qformer_files:
                raise RuntimeError("No QFormer keys found in {}".format(index_filename))

            logger.info("QFormer checkpoint shards selected from %s: %s", repo_id, qformer_files)
            for shard in qformer_files:
                path = hf_hub_download(repo_id, shard, local_files_only=local_files_only)
                yield from Blip2Base.iter_checkpoint_file_tensors(path)
            return

        for filename in ("model.safetensors", "pytorch_model.bin"):
            try:
                path = hf_hub_download(repo_id, filename, local_files_only=local_files_only)
            except Exception as exc:
                last_error = exc
                continue
            yield from Blip2Base.iter_checkpoint_file_tensors(path)
            return

        raise RuntimeError("Could not resolve Hugging Face checkpoint {}: {}".format(repo_id, last_error))

    @staticmethod
    def iter_checkpoint_file_tensors(path):
        if path.endswith(".safetensors"):
            from safetensors import safe_open

            with safe_open(path, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key == "query_tokens" or ".query_tokens" in key or "qformer." in key.lower():
                        yield key, handle.get_tensor(key)
            return

        checkpoint = torch.load(path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(checkpoint["model"], dict):
            checkpoint = checkpoint["model"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            checkpoint = checkpoint["state_dict"]
        if not isinstance(checkpoint, dict):
            raise TypeError("Unsupported checkpoint type from {}: {}".format(path, type(checkpoint)))

        for key, tensor in checkpoint.items():
            if torch.is_tensor(tensor) and (key == "query_tokens" or ".query_tokens" in key or "qformer." in key.lower()):
                yield key, tensor

#     def init_vision_encoder(
#         self, model_name, img_size, drop_path_rate, use_grad_checkpoint, precision
#     ):
#         assert model_name in [
#             "eva_clip_g",
#             "eva2_clip_L",
#             "clip_L",
#         ], "vit model must be eva_clip_g, eva2_clip_L or clip_L"
#         if model_name == "eva_clip_g":
#             visual_encoder = create_eva_vit_g(
#                 img_size, drop_path_rate, use_grad_checkpoint, precision
#             )
# #         elif model_name == "eva2_clip_L":
# #             visual_encoder = create_eva2_vit_L(
# #                 img_size, drop_path_rate, use_grad_checkpoint, precision
# #             )
#         elif model_name == "clip_L":
#             visual_encoder = create_clip_vit_L(img_size, use_grad_checkpoint, precision)
#         ln_vision = LayerNorm(visual_encoder.num_features)
#         self.vit_name = model_name
#         return visual_encoder, ln_vision

    # def load_from_pretrained(self, url_or_filename):
    #     if is_url(url_or_filename):
    #         cached_file = download_cached_file(
    #             url_or_filename, check_hash=False, progress=True
    #         )
    #         checkpoint = torch.load(cached_file, map_location="cpu")
    #     elif os.path.isfile(url_or_filename):
    #         checkpoint = torch.load(url_or_filename, map_location="cpu")
    #     else:
    #         raise RuntimeError("checkpoint url or path is invalid")

    #     state_dict = checkpoint["model"]

    #     msg = self.load_state_dict(state_dict, strict=False)

    #     # logging.info("Missing keys {}".format(msg.missing_keys))
    #     logging.info("load checkpoint from %s" % url_or_filename)

    #     return msg

    # def get_optimizer_params(self, weight_decay, lr_scale=1):
    #     if self.vit_name == "eva_clip_g":
    #         vit_num_layers = self.visual_encoder.get_num_layer()
    #         lr_scales = list(lr_scale ** (vit_num_layers + 1 - i) for i in range(vit_num_layers + 2))

    #         parameter_group_names = {}
    #         parameter_group_vars = {}

    #         for name, param in self.named_parameters():
    #             if not param.requires_grad:
    #                 continue  # frozen weights
    #             if len(param.shape) == 1 or name.endswith(".bias"):
    #                 group_name = "no_decay"
    #                 this_weight_decay = 0.
    #             else:
    #                 group_name = "decay"
    #                 this_weight_decay = weight_decay
    #             if 'visual_encoder' in name:
    #                 layer_id = self.visual_encoder.get_num_layer(name.replace('visual_encoder.',''))
    #                 group_name = "vit_layer_%d_%s" % (layer_id, group_name)
    #             else:
    #                 layer_id = None

    #             if group_name not in parameter_group_names:
    #                 if layer_id is not None:
    #                     scale = lr_scales[layer_id]
    #                 else:
    #                     scale = 1
    #                 parameter_group_names[group_name] = {
    #                     "weight_decay": this_weight_decay,
    #                     "params": [],
    #                     "lr_scale": scale
    #                 }
    #                 parameter_group_vars[group_name] = {
    #                     "weight_decay": this_weight_decay,
    #                     "params": [],
    #                     "lr_scale": scale
    #                 }
    #             parameter_group_vars[group_name]["params"].append(param)
    #             parameter_group_names[group_name]["params"].append(name)
    #         # import json
    #         # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    #         optim_params = list(parameter_group_vars.values())
    #         return optim_params
    #     else:
    #         return super().get_optimizer_params(weight_decay,lr_scale)

    # def _lemmatize(self, answers):
    #     def apply(answer):
    #         doc = self.lemmatizer(answer)

    #         words = []
    #         for token in doc:
    #             if token.pos_ in ["NOUN", "VERB"]:
    #                 words.append(token.lemma_)
    #             else:
    #                 words.append(token.text)
    #         answer = " ".join(words)

    #         return answer

    #     return [apply(answer) for answer in answers]

    # @property
    # def lemmatizer(self):
        # if self._lemmatizer is None:
        #     try:
        #         import spacy

        #         self._lemmatizer = spacy.load("en_core_web_sm")
        #     except ImportError:
        #         logging.error(
        #             """
        #             Please install spacy and en_core_web_sm model to apply lemmatization.
        #             python -m spacy download en_core_web_sm
        #             OR
        #             import spacy.cli
        #             spacy.cli.download("en_core_web_sm")
        #             """
        #         )
        #         exit(1)

        # return self._lemmatizer

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


# class LayerNorm(nn.LayerNorm):
#     """Subclass torch's LayerNorm to handle fp16."""

#     def forward(self, x: torch.Tensor):
#         orig_type = x.dtype
#         ret = super().forward(x.type(torch.float32))
#         return ret.type(orig_type)


# def compute_sim_matrix(model, data_loader, **kwargs):
#     k_test = kwargs.pop("k_test")

#     metric_logger = MetricLogger(delimiter="  ")
#     header = "Evaluation:"

#     logging.info("Computing features for evaluation...")
#     start_time = time.time()

#     texts = data_loader.dataset.text
#     num_text = len(texts)
#     text_bs = 256
#     text_ids = []
#     text_embeds = []
#     text_atts = []
#     for i in range(0, num_text, text_bs):
#         text = texts[i : min(num_text, i + text_bs)]
#         text_input = model.tokenizer(
#             text,
#             padding="max_length",
#             truncation=True,
#             max_length=35,
#             return_tensors="pt",
#         ).to(model.device)
#         text_feat = model.forward_text(text_input)
#         text_embed = F.normalize(model.text_proj(text_feat))
#         text_embeds.append(text_embed)
#         text_ids.append(text_input.input_ids)
#         text_atts.append(text_input.attention_mask)

#     text_embeds = torch.cat(text_embeds, dim=0)
#     text_ids = torch.cat(text_ids, dim=0)
#     text_atts = torch.cat(text_atts, dim=0)

#     vit_feats = []
#     image_embeds = []
#     for samples in data_loader:
#         image = samples["image"]

#         image = image.to(model.device)
#         image_feat, vit_feat = model.forward_image(image)
#         image_embed = model.vision_proj(image_feat)
#         image_embed = F.normalize(image_embed, dim=-1)

#         vit_feats.append(vit_feat.cpu())
#         image_embeds.append(image_embed)

#     vit_feats = torch.cat(vit_feats, dim=0)
#     image_embeds = torch.cat(image_embeds, dim=0)

#     sims_matrix = []
#     for image_embed in image_embeds:
#         sim_q2t = image_embed @ text_embeds.t()
#         sim_i2t, _ = sim_q2t.max(0)
#         sims_matrix.append(sim_i2t)
#     sims_matrix = torch.stack(sims_matrix, dim=0)

#     score_matrix_i2t = torch.full(
#         (len(data_loader.dataset.image), len(texts)), -100.0
#     ).to(model.device)

#     num_tasks = dist_utils.get_world_size()
#     rank = dist_utils.get_rank()
#     step = sims_matrix.size(0) // num_tasks + 1
#     start = rank * step
#     end = min(sims_matrix.size(0), start + step)

#     for i, sims in enumerate(
#         metric_logger.log_every(sims_matrix[start:end], 50, header)
#     ):
#         topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
#         image_inputs = vit_feats[start + i].repeat(k_test, 1, 1).to(model.device)
#         score = model.compute_itm(
#             image_inputs=image_inputs,
#             text_ids=text_ids[topk_idx],
#             text_atts=text_atts[topk_idx],
#         ).float()
#         score_matrix_i2t[start + i, topk_idx] = score + topk_sim

#     sims_matrix = sims_matrix.t()
#     score_matrix_t2i = torch.full(
#         (len(texts), len(data_loader.dataset.image)), -100.0
#     ).to(model.device)

#     step = sims_matrix.size(0) // num_tasks + 1
#     start = rank * step
#     end = min(sims_matrix.size(0), start + step)

#     for i, sims in enumerate(
#         metric_logger.log_every(sims_matrix[start:end], 50, header)
#     ):
#         topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
#         image_inputs = vit_feats[topk_idx.cpu()].to(model.device)
#         score = model.compute_itm(
#             image_inputs=image_inputs,
#             text_ids=text_ids[start + i].repeat(k_test, 1),
#             text_atts=text_atts[start + i].repeat(k_test, 1),
#         ).float()
#         score_matrix_t2i[start + i, topk_idx] = score + topk_sim

#     if dist_utils.is_dist_avail_and_initialized():
#         dist.barrier()
#         torch.distributed.all_reduce(
#             score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
#         )
#         torch.distributed.all_reduce(
#             score_matrix_t2i, op=torch.distributed.ReduceOp.SUM
#         )

#     total_time = time.time() - start_time
#     total_time_str = str(datetime.timedelta(seconds=int(total_time)))
#     logging.info("Evaluation time {}".format(total_time_str))

#     return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()
