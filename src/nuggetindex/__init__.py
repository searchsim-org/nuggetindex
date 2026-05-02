"""nuggetindex - governed atomic-fact index for RAG."""

from nuggetindex._version import __version__
from nuggetindex.adapters import (
    CorpusSource,
    HaystackCorpus,
    JsonlCorpus,
    LlamaIndexCorpus,
    ProxyPool,
    VespaCorpus,
    WebSearchCorpus,
)
from nuggetindex.audit.api import (
    AuditReport,
    ConflictRecord,
    StaleRecord,
    audit,
    audit_batch,
)
from nuggetindex.audit.discover import (
    PredicateProposal,
    SchemaProposal,
    discover_schema,
    merge_proposal,
)
from nuggetindex.audit.doctor import DoctorReport, DoctorScore, scan_index
from nuggetindex.audit.seeds import SeedCandidate, SeedProposal, propose_seeds
from nuggetindex.auto import AutoReport, auto
from nuggetindex.chains import ChainEdge, ChainEdgeType, NuggetChain
from nuggetindex.core.enums import EpistemicRank, LifecycleStatus, NuggetKind
from nuggetindex.core.errors import (
    BackendUnavailable,
    ChainAmbiguousError,
    ChainCycleDetected,
    ChainDepthExceeded,
    ConflictUnresolved,
    ExtractionFailed,
    InvalidRelationSchema,
    JudgeTimeout,
    NuggetIndexError,
)
from nuggetindex.core.models import (
    EpistemicState,
    FactTriple,
    Nugget,
    ProvenanceRecord,
    ValidityInterval,
)
from nuggetindex.core.schema import RelationKind, RelationSchema
from nuggetindex.extractors import CachedExtractor, TriggerExtractor
from nuggetindex.sidecar import (
    ContextFormatter,
    JITPassageCache,
    Router,
    Sidecar,
    SidecarResponse,
)
from nuggetindex.store import AddResult, IngestResult, NuggetStore

__all__ = [
    "__version__",
    "AddResult",
    "AuditReport",
    "AutoReport",
    "BackendUnavailable",
    "CachedExtractor",
    "ChainAmbiguousError",
    "ChainCycleDetected",
    "ChainDepthExceeded",
    "ChainEdge",
    "ChainEdgeType",
    "ConflictRecord",
    "ConflictUnresolved",
    "ContextFormatter",
    "CorpusSource",
    "DoctorReport",
    "DoctorScore",
    "EpistemicRank",
    "EpistemicState",
    "ExtractionFailed",
    "FactTriple",
    "HaystackCorpus",
    "IngestResult",
    "InvalidRelationSchema",
    "JITPassageCache",
    "JsonlCorpus",
    "JudgeTimeout",
    "LifecycleStatus",
    "LlamaIndexCorpus",
    "Nugget",
    "NuggetChain",
    "NuggetIndexError",
    "NuggetKind",
    "NuggetStore",
    "PredicateProposal",
    "ProvenanceRecord",
    "ProxyPool",
    "RelationKind",
    "RelationSchema",
    "Router",
    "SchemaProposal",
    "SeedCandidate",
    "SeedProposal",
    "Sidecar",
    "SidecarResponse",
    "StaleRecord",
    "TriggerExtractor",
    "ValidityInterval",
    "VespaCorpus",
    "WebSearchCorpus",
    "audit",
    "audit_batch",
    "auto",
    "discover_schema",
    "merge_proposal",
    "propose_seeds",
    "scan_index",
]
