import torch
from torch import nn
from transformers import T5ForConditionalGeneration

try:
    from peft import LoraConfig, TaskType, get_peft_model
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


class FlanT5Decoder(nn.Module):
    def __init__(
        self,
        model_name="google/flan-t5-base",
        encoder_hidden_size=768,
        use_lora=True,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
    ):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.hidden_size = self.model.config.d_model
        self.vocab_size = self.model.config.vocab_size

        self.encoder_proj = nn.Identity()
        if encoder_hidden_size != self.hidden_size:
            self.encoder_proj = nn.Linear(encoder_hidden_size, self.hidden_size)

        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = 0

        if use_lora:
            if not PEFT_AVAILABLE:
                raise ImportError("peft is required when use_lora=True. Please install peft.")
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q", "v"],
                bias="none",
            )
            self.model = get_peft_model(self.model, lora_config)

    def _build_labels(self, labels, answer_mask):
        if labels is None:
            return None
        labels = labels.clone()
        if answer_mask is not None:
            labels = labels.masked_fill(answer_mask == 0, -100)
        labels = labels.masked_fill(labels < 0, -100)
        return labels

    def forward(self, input_ids, encoder_outs, answer_mask=None, encoder_mask=None, labels=None):
        encoder_outs = self.encoder_proj(encoder_outs)
        labels = self._build_labels(labels, answer_mask)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=encoder_mask,
            encoder_outputs=(encoder_outs,),
            decoder_attention_mask=answer_mask,
            labels=labels,
            return_dict=True,
        )
        return outputs.logits, outputs.loss
