"""Named-entity recognition (optional; English-centric).

Honest scope: off-the-shelf NER is trained on news/web text. WhatsApp chat is informal, lower-cased,
code-mixed and full of nicknames, so precision is low and it often misses Roman-Urdu/Hinglish
names. Treat counts as a rough exploratory signal, not a measurement. Entities are *people's names*,
i.e. personal data: results are only aggregated counts and are masked when anonymisation is on."""
from __future__ import annotations

from collections import Counter

import pandas as pd

DEFAULT_LABELS = {"PERSON", "ORG", "GPE", "LOC", "DATE", "EVENT", "PRODUCT", "NORP", "FAC"}


def load_spacy(model: str = "en_core_web_sm"):
    try:
        import spacy
        return spacy.load(model, disable=["lemmatizer"])
    except Exception as exc:        # noqa: BLE001
        raise ImportError(f"spaCy model '{model}' unavailable ({exc}). "
                          "pip install spacy && python -m spacy download en_core_web_sm") from exc


def extract_entities(texts: list[str], nlp=None, labels: set[str] | None = None, top: int = 30,
                     batch_size: int = 256) -> pd.DataFrame:
    labels = labels or DEFAULT_LABELS
    nlp = nlp or load_spacy()
    c: Counter = Counter()
    for doc in nlp.pipe(texts, batch_size=batch_size):
        for ent in doc.ents:
            if ent.label_ in labels:
                c[(ent.text.strip(), ent.label_)] += 1
    rows = [{"entity": e, "label": lab, "count": n} for (e, lab), n in c.most_common(top)]
    return pd.DataFrame(rows, columns=["entity", "label", "count"])


def hf_ner_pipeline(model_id: str = "Davlan/xlm-roberta-base-ner-hrl"):
    """Multilingual transformer NER (Colab). Returns a callable like a spaCy ``pipe`` adapter."""
    import torch
    from transformers import pipeline
    p = pipeline("token-classification", model=model_id, aggregation_strategy="simple",
                 device=0 if torch.cuda.is_available() else -1)

    class _Adapter:
        def pipe(self, texts, batch_size=64):
            for _t, ents in zip(texts, p(list(texts), batch_size=batch_size), strict=True):
                yield type("Doc", (), {"ents": [type("E", (), {"text": e["word"], "label_": e["entity_group"]})
                                                for e in ents]})()
    return _Adapter()
