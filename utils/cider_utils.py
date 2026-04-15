"""CIDEr scorer with pre-computed corpus-level IDF statistics.

Standard CIDEr recomputes IDF from the references in each batch, which
is noisy and unreliable for small batches (e.g., during SCST training).
This module pre-computes IDF from the entire training corpus once, then
uses those fixed statistics for all subsequent CIDEr computations.
"""

import logging
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class CorpusCider:
    """CIDEr scorer that uses corpus-level IDF instead of batch-level IDF.

    Usage:
        scorer = CorpusCider()
        scorer.init_corpus_df(video_sentences_dict)  # pre-compute once
        score, scores = scorer.compute_score(gts, res)  # same API as Cider
    """

    def __init__(self, n=4, sigma=6.0):
        self._n = n
        self._sigma = sigma
        self._corpus_df = None
        self._corpus_ref_len = None

    def init_corpus_df(self, video_sentences_dict):
        """Pre-compute document frequency from all training references.

        Args:
            video_sentences_dict: dict {video_id: [caption_str, ...]}
                All reference captions for each video in the training set.
        """
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
        from pycocoevalcap.cider.cider_scorer import CiderScorer

        # Build PTBTokenizer-compatible format
        refs_dict = {}
        for idx, (vid, captions) in enumerate(video_sentences_dict.items()):
            refs_dict[idx] = [{'caption': c} for c in captions]

        # PTB-tokenize all references (same tokenization used during scoring)
        ptb = PTBTokenizer()
        tokenized = ptb.tokenize(refs_dict)

        # Feed to CiderScorer to compute corpus-level document frequency
        scorer = CiderScorer(n=self._n, sigma=self._sigma)
        for idx in sorted(tokenized.keys()):
            scorer += (None, tokenized[idx])
        scorer.compute_doc_freq()

        self._corpus_df = dict(scorer.document_frequency)
        # ref_len = log(N) where N = number of documents (videos) in corpus
        self._corpus_ref_len = np.log(float(len(scorer.crefs)))

        total_refs = sum(len(v) for v in video_sentences_dict.values())
        logger.info(
            "CorpusCider: pre-computed IDF from %d videos, %d total refs, %d unique n-grams",
            len(video_sentences_dict), total_refs, len(self._corpus_df)
        )

    def compute_score(self, gts, res):
        """Compute CIDEr score using pre-computed corpus-level IDF.

        Same interface as pycocoevalcap.cider.cider.Cider.compute_score().

        Args:
            gts: dict {id: [ref_str, ...]} — ground truth references (PTB-tokenized)
            res: dict {id: [hyp_str]} — hypothesis (PTB-tokenized)

        Returns:
            (mean_score, scores_array)
        """
        from pycocoevalcap.cider.cider_scorer import CiderScorer

        scorer = CiderScorer(n=self._n, sigma=self._sigma)
        for id in sorted(gts.keys()):
            scorer += (res[id][0], gts[id])

        if self._corpus_df is not None:
            # Inject pre-computed corpus-level DF instead of batch-level DF
            scorer.document_frequency = self._corpus_df
            scorer.ref_len = self._corpus_ref_len
            score = scorer.compute_cider()
            return np.mean(np.array(score)), np.array(score)
        else:
            # Fallback to standard batch-level computation
            return scorer.compute_score()
