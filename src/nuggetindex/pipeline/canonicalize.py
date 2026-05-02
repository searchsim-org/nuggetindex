"""Stage 2: subject normalization + predicate canonicalization.

Pure function. Returns a new ``Nugget`` (the core model is frozen) with:

* leading/trailing whitespace stripped from the subject,
* predicate mapped to its canonical form via the schema's alias table,
* content-hashed ID recomputed (since subject/predicate are part of the hash).

Alias lookup for subjects (entity resolution) is out of scope for v0.1; the
subject is normalized by whitespace only so that callers get stable keys when
evidence text is noisy.
"""

from __future__ import annotations

from nuggetindex.core.models import FactTriple, Nugget
from nuggetindex.core.schema import RelationSchema


def canonicalize(nugget: Nugget, schema: RelationSchema) -> Nugget:
    """Return a new ``Nugget`` with normalized subject and canonical predicate.

    The nugget ID is content-hashed over ``(subject, predicate, object,
    validity.start, scope)``, so changing either field changes the ID. We
    therefore rebuild via ``Nugget.new`` rather than ``model_copy``.
    """
    subject = nugget.fact.subject.strip()
    predicate = schema.canonicalize(nugget.fact.predicate)

    new_fact = FactTriple(
        subject=subject,
        predicate=predicate,
        object=nugget.fact.object,
        text=nugget.fact.text,
        # Preserve LLM-emitted entity types through canonicalization so
        # downstream stages (fix 9) can use them instead of falling back
        # to spaCy NER.
        subject_type=nugget.fact.subject_type,
        object_type=nugget.fact.object_type,
    )
    return Nugget.new(
        kind=nugget.kind,
        fact=new_fact,
        validity=nugget.validity,
        epistemic=nugget.epistemic,
        provenance=nugget.provenance,
        parent_id=nugget.parent_id,
        extraction_confidence=nugget.extraction_confidence,
        created_at=nugget.created_at,
        updated_at=nugget.updated_at,
    )
