torchrun --nproc_per_node=2 --standalone \
main_task_caption.py \
--do_train --stage_two --task_type caption --datatype msrvtt \
--num_thread_reader=4 \
--epochs=5 --batch_size=64 \
--n_display=50 \
--train_csv data/msrvtt/MSRVTT_train.9k.csv \
--val_csv data/msrvtt/MSRVTT_JSFUSION_test.csv \
--data_path data/msrvtt/MSRVTT_data.json \
--features_path data/msrvtt/msrvtt_videos_features.pickle \
--output_dir ckpts/ckpt_msrvtt_caption --bert_model bert-base-uncased \
--do_lower_case --lr 3e-5 --max_words 48 --max_frames 48 \
--batch_size_val 32 --visual_num_hidden_layers 6 \
--freeze_vit \
--init_model weight/univl.pretrained.bin \
--video_dim 768 \
--gradient_accumulation_steps=2