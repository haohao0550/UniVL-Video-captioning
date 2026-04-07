"""
Manage beam search info structure.
Heavily borrowed from OpenNMT-py.
For code in OpenNMT-py, please check the following link (maybe in oldest version):
https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/Beam.py
"""

import torch

class Constants():
    def __init__(self):
        self.PAD = 0
        self.UNK = 1
        self.BOS = 2
        self.EOS = 3
        self.PAD_WORD = '[PAD]'
        self.UNK_WORD = '[UNK]'
        self.BOS_WORD = '[CLS]'
        self.EOS_WORD = '[SEP]'

    @classmethod
    def from_tokenizer(cls, tokenizer):
        instance = cls()
        if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
            instance.PAD = int(tokenizer.pad_token_id)
        elif hasattr(tokenizer, "vocab") and instance.PAD_WORD in tokenizer.vocab:
            instance.PAD = tokenizer.vocab[instance.PAD_WORD]

        if hasattr(tokenizer, "unk_token_id") and tokenizer.unk_token_id is not None:
            instance.UNK = int(tokenizer.unk_token_id)
        elif hasattr(tokenizer, "vocab") and instance.UNK_WORD in tokenizer.vocab:
            instance.UNK = tokenizer.vocab[instance.UNK_WORD]

        if hasattr(tokenizer, "cls_token_id") and tokenizer.cls_token_id is not None:
            instance.BOS = int(tokenizer.cls_token_id)
        elif hasattr(tokenizer, "bos_token_id") and tokenizer.bos_token_id is not None:
            instance.BOS = int(tokenizer.bos_token_id)
        elif hasattr(tokenizer, "vocab") and instance.BOS_WORD in tokenizer.vocab:
            instance.BOS = tokenizer.vocab[instance.BOS_WORD]
        else:
            instance.BOS = instance.PAD

        if hasattr(tokenizer, "sep_token_id") and tokenizer.sep_token_id is not None:
            instance.EOS = int(tokenizer.sep_token_id)
        elif hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            instance.EOS = int(tokenizer.eos_token_id)
        elif hasattr(tokenizer, "vocab") and instance.EOS_WORD in tokenizer.vocab:
            instance.EOS = tokenizer.vocab[instance.EOS_WORD]
        else:
            instance.EOS = instance.PAD
        return instance

class Beam():
    ''' Beam search '''

    def __init__(self, size, device=False, tokenizer=None):
        if tokenizer is None:
            self.constants = Constants()
        else:
            self.constants = Constants.from_tokenizer(tokenizer)

        self.size = size
        self._done = False
        # The score for each interface on the beam.
        self.scores = torch.zeros((size,), dtype=torch.float, device=device)
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [torch.full((size,), self.constants.BOS, dtype=torch.long, device=device)]

    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    @property
    def done(self):
        return self._done

    def advance(self, word_prob, word_length=None):

        "Update beam status and check if finished or not."
        num_words = word_prob.size(1)
        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_lk = word_prob + self.scores.unsqueeze(1).expand_as(word_prob)
        else:
            beam_lk = word_prob[0]
        flat_beam_lk = beam_lk.view(-1)
        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True) # 1st sort
        self.all_scores.append(self.scores)
        self.scores = best_scores
        # bestScoresId is flattened as a (beam x word) array,
        # so we need to calculate which word and beam each score came from
        prev_k = best_scores_id // num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append(best_scores_id - prev_k * num_words)
        # End condition is when top-of-beam is EOS.
        if self.next_ys[-1][0].item() == self.constants.EOS:
            self._done = True

        return self._done

    def sort_scores(self):
        "Sort the scores."
        return torch.sort(self.scores, 0, True)

    def get_the_best_score_and_idx(self):
        "Get the score of the best in the beam."
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self):
        "Get the decoded sequence for the current timestep."

        if len(self.next_ys) == 1:
            dec_seq = self.next_ys[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()
            hyps = [self.get_hypothesis(k) for k in keys]
            hyps = [[self.constants.BOS] + h for h in hyps]
            dec_seq = torch.LongTensor(hyps)

        return dec_seq

    def get_hypothesis(self, k):
        """ Walk back to construct the full hypothesis. """
        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_ys[j+1][k])
            k = self.prev_ks[j][k]

        return list(map(lambda x: x.item(), hyp[::-1]))
