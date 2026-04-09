import torch
from inference.eval_utils import decode_tokens_to_text, save_predictions, save_complete_results, log_metrics


def eval_epoch(args, model, test_dataloader, tokenizer, device, n_gpu, logger, nlgEvalObj=None, test_set=None):
    if hasattr(model, 'module'):
        model = model.module.to(device)

    if model._stage_one:
        return 0.

    all_result_lists = []
    all_caption_lists = []
    total_loss = 0.0
    model.eval()
    for batch in test_dataloader:
        batch = tuple(t.to(device, non_blocking=True) for t in batch)

        input_ids, input_mask, segment_ids, video, video_mask, \
        pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
        pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids = batch

        with torch.no_grad():
            # Calculate validation loss
            # forward() returns (loss, visual_output) in eval mode,
            # so we can reuse visual_output for generation without re-encoding.
            loss, visual_output = model(input_ids, segment_ids, input_mask, video, video_mask,
                        pairs_masked_text=pairs_masked_text, pairs_token_labels=pairs_token_labels,
                        masked_video=masked_video, video_labels_index=video_labels_index,
                        input_caption_ids=pairs_input_caption_ids, decoder_mask=pairs_decoder_mask,
                        output_caption_ids=pairs_output_caption_ids)
            if loss is not None:
                if n_gpu > 1:
                    loss = loss.mean()
                total_loss += float(loss)

            video_mask = video_mask.view(-1, video_mask.shape[-1])

            beam_size = max(1, getattr(model, "beam_size", 1))
            max_length = getattr(model, "max_txt_len", args.max_words)
            generated_ids = model.generate_caption_ids(
                visual_output, video_mask, num_beams=beam_size, max_length=max_length
            )
            all_result_lists.extend(model.t5_tokenizer.batch_decode(generated_ids, skip_special_tokens=True))

            pairs_output_caption_ids = pairs_output_caption_ids.view(-1, pairs_output_caption_ids.shape[-1])
            caption_list = pairs_output_caption_ids.cpu().detach().numpy()

            for re_idx, re_list in enumerate(caption_list):
                decode_text = decode_tokens_to_text(re_list, tokenizer)
                all_caption_lists.append(decode_text)

    # Calculate and log average validation loss
    avg_val_loss = total_loss / len(test_dataloader)
    logger.info("  Average Validation Loss: {:.4f}".format(avg_val_loss)) 
    
    complete_results_path = save_complete_results(all_result_lists, test_set, args.output_dir)
    if complete_results_path:
        logger.info("File of complete results is saved in {}".format(complete_results_path))

    save_predictions(all_result_lists, all_caption_lists, args.output_dir)

    if args.datatype == "msrvtt":
        all_caption_lists = []
        sentences_dict = test_dataloader.dataset.sentences_dict
        video_sentences_dict = test_dataloader.dataset.video_sentences_dict
        for idx in range(len(sentences_dict)):
            video_id, _ = sentences_dict[idx]
            sentences = video_sentences_dict[video_id]
            all_caption_lists.append(sentences)
        all_caption_lists = [list(itms) for itms in zip(*all_caption_lists)]
    else:
        all_caption_lists = [all_caption_lists]

    if nlgEvalObj is not None:
        metrics_nlg = nlgEvalObj.compute_metrics(ref_list=all_caption_lists, hyp_list=all_result_lists)
        log_metrics(logger, metrics_nlg)
        Bleu_4 = metrics_nlg["Bleu_4"]
    else:
        logger.warning("Evaluation metrics skipped (pycocoevalcap not available)")
        Bleu_4 = 0.0
    
    return Bleu_4, avg_val_loss
