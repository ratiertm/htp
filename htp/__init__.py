"""
HTP  -  Hub Topology Programming
"""

# ── Phase 1: 단일 영역 ──────────────────────────────
from .runtime.htp_runtime import (
    HTPConfig,
    WeightMatrix,
    HubFormationEngine,
    PruningEngine,
    ActivationEngine,
    HTPRuntime,
    Node,
    RunResult,
    tag,
    terminal,
    FIRE_FLOOR,
)

# ── Phase 2: 다중 영역 + Thalamus ───────────────────
from .runtime.region_runtime       import RegionRuntime
from .runtime.brain_runtime        import PFCRuntime, BrainRuntime
from .runtime.cortical_connections import CorticalConnections

from .thalamus import (
    RegionSignal,
    GatingMask,
    CompetitionResult,
    ThalamusOutput,
    Action,
    CoreCells,
    MatrixCells,
    NGETrigger,
    Thalamus,
    TopDownSignal,
    TopDownBias,
)

# ── Phase 5: Memory System ─────────────────────────
from .memory import (
    MemorySystem,
    Episode,
    Pattern,
    MemoryContext,
)


# ── Phase 4: LLM-as-Node + Async ────────────────────
# sub-4 (2026-05-19): LLMRegion + ExternalRegion 신규.
# LLMRegionRuntime → archive/deprecated_phase4/ (Plan FR-20, C-4 isolated 감소).
from .llm import (
    LLMNode, MockLLMNode, CostRouter,
    LLMRegion, SPECIALTY_PROMPTS,
)
from .runtime.external_region    import ExternalRegion
from .runtime.async_brain_runtime import AsyncBrainRuntime
from .runtime.pipelined_brain     import PipelinedBrainRuntime

__all__ = [
    # Phase 1
    "HTPConfig",
    "WeightMatrix",
    "HubFormationEngine",
    "PruningEngine",
    "ActivationEngine",
    "HTPRuntime",
    "Node",
    "RunResult",
    "tag",
    "terminal",
    "FIRE_FLOOR",
    # Phase 2
    "RegionRuntime",
    "PFCRuntime",
    "BrainRuntime",
    "RegionSignal",
    "GatingMask",
    "CompetitionResult",
    "ThalamusOutput",
    "Action",
    "CoreCells",
    "MatrixCells",
    "NGETrigger",
    "Thalamus",
    # Phase 3
    "TopDownSignal",
    "TopDownBias",
    "CorticalConnections",
    # Phase 4
    "LLMNode",
    "MockLLMNode",
    "CostRouter",
    "LLMRegion",
    "SPECIALTY_PROMPTS",
    "ExternalRegion",
    "AsyncBrainRuntime",
    "PipelinedBrainRuntime",
    # Phase 5
    "MemorySystem",
    "Episode",
    "Pattern",
    "MemoryContext",
]
