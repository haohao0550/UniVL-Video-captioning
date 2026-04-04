#!/usr/bin/env bash

# ------------------------------------------------------------
# 1) RESEARCH RUN (fixed prompt + ITC/ITM, two-path training)
# ------------------------------------------------------------
# torchrun --nproc_per_node=2 --standalone \
# main_task_caption_qformer.py \
# --do_train --num_thread_reader=4 \
# --epochs=30 --batch_size=64 \
# --n_display=50 \
# --train_csv data/msrvtt/MSRVTT_train.9k.csv \
# --val_csv data/msrvtt/MSRVTT_JSFUSION_test.csv \
# --data_path data/msrvtt/MSRVTT_data.json \
# --features_path /kaggle/input/datasets/phihungcrr1701/msrvtt-clip-vitl14-features/msrvtt_clip_vitl14_features.pickle \
# --output_dir ckpts/ckpt_msrvtt_caption_qformer --bert_model bert-base-uncased \
# --t5_model t5-base \
# --do_lower_case --lr 3e-5 --max_words 48 --max_frames 48 \
# --batch_size_val 32 --visual_num_hidden_layers 6 \
# --datatype msrvtt --stage_two \
# --init_model /kaggle/input/datasets/phihungcrr1701/best-model-t5-stage-1/pytorch_model.bin.8 \
# --gradient_accumulation_steps=2 \
# --video_dim 768 \
# --caption_prompt_mode fixed \
# --caption_prompt_text "What does the video describe?" \
# --freeze_epochs 5 \
# --use_itc_loss --itc_weight 0.03 --itc_proj_dim 256 --itc_init_temp 0.07 \
# --use_itm_loss --itm_weight 0.02 --itm_use_hard_negative \
# --it_aux_warmup_epochs 6 --it_aux_ramp_epochs 4

# ------------------------------------------------------------
# 2) BASELINE EMPTY PROMPT (video-only prompt path + ITC/ITM)
# ------------------------------------------------------------
torchrun --nproc_per_node=2 --standalone \
main_task_caption_qformer.py \
--do_train --num_thread_reader=4 \
--epochs=30 --batch_size=64 \
--n_display=50 \
--train_csv data/msrvtt/MSRVTT_train.9k.csv \
--val_csv data/msrvtt/MSRVTT_JSFUSION_test.csv \
--data_path data/msrvtt/MSRVTT_data.json \
--features_path /kaggle/input/datasets/phihungcrr1701/msrvtt-clip-vitl14-features/msrvtt_clip_vitl14_features.pickle \
--output_dir ckpts/ckpt_msrvtt_caption_qformer_empty_prompt --bert_model bert-base-uncased \
--t5_model t5-base \
--do_lower_case --lr 3e-5 --max_words 48 --max_frames 48 \
--batch_size_val 32 --visual_num_hidden_layers 6 \
--datatype msrvtt --stage_two \
--init_model /kaggle/input/datasets/phihungcrr1701/best-model-t5-stage-1/pytorch_model.bin.8 \
--gradient_accumulation_steps=2 \
--video_dim 768 \
--caption_prompt_mode empty \
--freeze_epochs 5 \
--use_itc_loss --itc_weight 0.03 --itc_proj_dim 256 --itc_init_temp 0.07 \
--use_itm_loss --itm_weight 0.02 --itm_use_hard_negative \
--it_aux_warmup_epochs 6 --it_aux_ramp_epochs 4

# ------------------------------------------------------------
# 3) BASELINE CAPTION-ONLY (fixed prompt, ITC/ITM OFF)
# ------------------------------------------------------------
# torchrun --nproc_per_node=2 --standalone \
# main_task_caption_qformer.py \
# --do_train --num_thread_reader=4 \
# --epochs=30 --batch_size=64 \
# --n_display=50 \
# --train_csv data/msrvtt/MSRVTT_train.9k.csv \
# --val_csv data/msrvtt/MSRVTT_JSFUSION_test.csv \
# --data_path data/msrvtt/MSRVTT_data.json \
# --features_path /kaggle/input/datasets/phihungcrr1701/msrvtt-clip-vitl14-features/msrvtt_clip_vitl14_features.pickle \
# --output_dir ckpts/ckpt_msrvtt_caption_qformer_caption_only --bert_model bert-base-uncased \
# --t5_model t5-base \
# --do_lower_case --lr 3e-5 --max_words 48 --max_frames 48 \
# --batch_size_val 32 --visual_num_hidden_layers 6 \
# --datatype msrvtt --stage_two \
# --init_model /kaggle/input/datasets/phihungcrr1701/best-model-t5-stage-1/pytorch_model.bin.8 \
# --gradient_accumulation_steps=2 \
# --video_dim 768 \
# --caption_prompt_mode fixed \
# --caption_prompt_text "What does the video describe?" \
# --freeze_epochs 5
