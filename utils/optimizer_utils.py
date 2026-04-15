import torch
from modules.optimization import BertAdam


def prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.):
    if hasattr(model, 'module'):
        model = model.module

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    def is_qformer_param(name):
        return name.startswith(("Qformer.", "query_tokens", "qformer_visual_proj."))

    def is_t5_decoder_param(name):
        return name.startswith(("t5_model.", "t5_proj."))

    def is_bert_param(name):
        return name.startswith("bert.")

    def is_other_param(name):
        return not (is_bert_param(name) or is_qformer_param(name) or is_t5_decoder_param(name))

    lr_qformer = getattr(args, "lr_qformer", args.lr)
    lr_lora = getattr(args, "lr_lora", args.lr)

    no_decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    no_decay_bert_param_tp = [(n, p) for n, p in no_decay_param_tp if is_bert_param(n)]
    no_decay_qformer_param_tp = [(n, p) for n, p in no_decay_param_tp if is_qformer_param(n)]
    no_decay_t5_decoder_param_tp = [(n, p) for n, p in no_decay_param_tp if is_t5_decoder_param(n)]
    no_decay_other_param_tp = [(n, p) for n, p in no_decay_param_tp if is_other_param(n)]

    decay_bert_param_tp = [(n, p) for n, p in decay_param_tp if is_bert_param(n)]
    decay_qformer_param_tp = [(n, p) for n, p in decay_param_tp if is_qformer_param(n)]
    decay_t5_decoder_param_tp = [(n, p) for n, p in decay_param_tp if is_t5_decoder_param(n)]
    decay_other_param_tp = [(n, p) for n, p in decay_param_tp if is_other_param(n)]

    optimizer_grouped_parameters = [
        {'params': [p for n, p in no_decay_bert_param_tp], 'weight_decay': 0.01, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in no_decay_qformer_param_tp], 'weight_decay': 0.01, 'lr': lr_qformer},
        {'params': [p for n, p in no_decay_t5_decoder_param_tp], 'weight_decay': 0.01, 'lr': lr_lora},
        {'params': [p for n, p in no_decay_other_param_tp], 'weight_decay': 0.01},
        {'params': [p for n, p in decay_bert_param_tp], 'weight_decay': 0.0, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in decay_qformer_param_tp], 'weight_decay': 0.0, 'lr': lr_qformer},
        {'params': [p for n, p in decay_t5_decoder_param_tp], 'weight_decay': 0.0, 'lr': lr_lora},
        {'params': [p for n, p in decay_other_param_tp], 'weight_decay': 0.0}
    ]

    # P2: Use constant LR schedule for SCST (RL) training instead of linear decay.
    # Linear decay drives LR to 0, which is counterproductive for RL fine-tuning.
    schedule = 'warmup_constant' if getattr(args, 'scst', False) else 'warmup_linear'

    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                         schedule=schedule, t_total=num_train_optimization_steps, weight_decay=0.01,
                         max_grad_norm=1.0)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=True)

    return optimizer, scheduler, model
