"""Toxicity scoring (optional, consent-gated, local inference only).

Privacy: the model weights are downloaded once and run on the local/Colab machine; message text is
never sent to a third-party API. Even so, flagging people's messages as "toxic" is sensitive, so the
scorer refuses to run without explicit consent and results are aggregated counts, not message
dumps. Models are imperfect and biased (e.g. against dialect, reclaimed slurs, banter): scores are
not moderation verdicts."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np


class ToxicityScorer:
    def __init__(self, consent: bool = False, model_id: str = "unitary/multilingual-toxic-xlm-roberta",
                 pipeline_factory: Callable | None = None, threshold: float = 0.7):
        if not consent:
            raise PermissionError("Toxicity analysis requires explicit user consent (consent=True).")
        self.model_id, self.threshold = model_id, threshold
        self._factory = pipeline_factory or self._default
        self._pipe = None

    def _default(self):
        import torch
        from transformers import pipeline
        return pipeline("text-classification", model=self.model_id, top_k=None, truncation=True,
                        max_length=128, device=0 if torch.cuda.is_available() else -1)

    def score(self, texts: list[str]) -> np.ndarray:
        """Probability of the toxic label per text."""
        if self._pipe is None:
            self._pipe = self._factory()
        out = []
        for scores in self._pipe(list(texts)):
            tox = [s["score"] for s in scores if "toxic" in str(s["label"]).lower()
                   and "non" not in str(s["label"]).lower()]
            out.append(max(tox) if tox else 0.0)
        return np.array(out)

    def flag(self, texts: list[str]) -> np.ndarray:
        return self.score(texts) >= self.threshold
