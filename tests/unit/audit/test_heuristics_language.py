"""Tests for language detection + ``stratify_by='language'`` (Fix 4)."""

from __future__ import annotations

import pytest

from nuggetindex.audit.heuristics import stratified_sample
from nuggetindex.audit.heuristics.language import _detect_language
from nuggetindex.pipeline.constructor import Document


def test_detect_language_latin() -> None:
    """Plain English prose should be detected as ``en`` (or at minimum not ``unk``)."""
    code = _detect_language("The sky is blue and the clouds are white today.")
    assert code != "unk"


def test_detect_language_cyrillic() -> None:
    """Cyrillic script should map to a language code, not ``unk``."""
    code = _detect_language("Большой привет миру из Москвы сегодня утром.")
    assert code != "unk"


def test_detect_language_empty_text_returns_unk() -> None:
    """Empty or whitespace-only input falls through to ``unk``."""
    assert _detect_language("") == "unk"
    assert _detect_language("   \n\t  ") == "unk"


@pytest.mark.asyncio
async def test_language_stratification_buckets_per_lang() -> None:
    """30 docs across 3 languages; a 12-doc sample should cover all 3 buckets."""
    pytest.importorskip("langdetect")

    eng = [
        "The quick brown fox jumps over the lazy dog near the river.",
        "A stitch in time saves nine when you handle laundry carefully.",
        "She sells seashells by the seashore every Sunday morning.",
        "All that glitters is not gold, even in this economy.",
        "The early bird catches the worm, the late one goes hungry.",
        "Practice makes perfect, or at least marginally acceptable.",
        "An apple a day keeps the doctor away from your doorstep.",
        "A journey of a thousand miles begins with a single step forward.",
        "Better late than never, but never late is better still.",
        "Do not judge a book by its glossy laminated cover alone.",
    ]
    rus = [
        "Большой привет миру из Москвы сегодня утром на улице.",
        "Без труда не выловишь и рыбку из пруда весной.",
        "Тише едешь — дальше будешь, говорит старая пословица народа.",
        "Не всё то золото, что блестит под ярким солнцем утром.",
        "Волков бояться — в лес не ходить никогда и никуда.",
        "Повторение — мать учения, особенно в изучении иностранных языков.",
        "Утро вечера мудренее, говорят старики в нашей деревне.",
        "Москва не сразу строилась, и Рим тоже не за один день.",
        "Друг познаётся в беде, а не в радости и веселье всегда.",
        "Семь раз отмерь, один раз отрежь, прежде чем начинать работу.",
    ]
    fra = [
        "Le chat noir dort paisiblement sur le canapé bleu du salon.",
        "Je mange du pain frais avec du beurre chaque matin au petit déjeuner.",
        "La vie est belle quand on sait apprécier les petits moments.",
        "Il pleut des cordes sur la ville de Paris aujourd'hui encore.",
        "Les enfants jouent dans le jardin du voisin tous les samedis.",
        "Le soleil brille fort au-dessus des montagnes enneigées ce matin.",
        "Nous allons au marché acheter des légumes frais pour la semaine.",
        "Elle écrit une longue lettre à sa grand-mère chaque dimanche après-midi.",
        "Le train part de la gare du nord à dix heures précises ce matin.",
        "Mon frère habite dans une petite maison près de la rivière.",
    ]
    docs: list[Document] = []
    for lang_label, texts in (("en", eng), ("ru", rus), ("fr", fra)):
        for i, t in enumerate(texts):
            docs.append(
                Document(
                    source_id=f"{lang_label}-{i:02d}",
                    text=t,
                    uri=None,
                    source_date=None,
                )
            )

    sampled, n_total = await stratified_sample(
        docs, sample_size=12, stratify_by="language", rng_seed=0
    )
    assert n_total == 30
    assert len(sampled) == 12

    detected = {_detect_language(d.text or "") for d in sampled}
    # Expect at least three distinct language buckets covered. We don't pin
    # exact labels because langdetect may identify French as 'fr' or similar
    # variants; we only require coverage.
    assert len(detected) >= 3, f"expected >=3 language buckets, got {sorted(detected)}"
