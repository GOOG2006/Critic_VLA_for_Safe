"""Safe-MoLe: routing-aware safety filtering for MoLe-VLA.

See DERIVATION_PACKAGE.md v4 for theoretical background.
"""
from .critic import SafetyCriticHead
from .projection import ISSfCBFProjection, issf_cbf_project, CBFParams
from .routing import SafetyAwareBudget, BudgetConfig
from .conformal import split_conformal_quantile, global_worst_case_quantile
from .integration import SafeMoLeExtension, SafeMoLeConfig
from .barriers import TableHeightBarrier, WorkspaceBoxBarrier, BarrierRegistry
from .cogact_adapter import attach_safe_mole_to, SafeCogACT

# Optional (require trimesh): SDF barrier
try:
    from .sdf import (
        MeshSDFBarrier, RLBenchSDFBarrier, VoxelizedSDFCache,
        make_box_mesh, make_sphere_mesh,
    )
    _HAS_SDF = True
except ImportError:
    _HAS_SDF = False

# Calibration helpers
from .calibration_collector import (
    ConformalDataCollector, CalibrationConfig,
    make_dataset_replay_iterator,
)

__all__ = [
    "SafetyCriticHead",
    "ISSfCBFProjection",
    "issf_cbf_project",
    "CBFParams",
    "SafetyAwareBudget",
    "BudgetConfig",
    "split_conformal_quantile",
    "global_worst_case_quantile",
    "SafeMoLeExtension",
    "SafeMoLeConfig",
    "TableHeightBarrier",
    "WorkspaceBoxBarrier",
    "BarrierRegistry",
    "attach_safe_mole_to",
    "SafeCogACT",
    "ConformalDataCollector",
    "CalibrationConfig",
    "make_dataset_replay_iterator",
]
if _HAS_SDF:
    __all__ += [
        "MeshSDFBarrier", "RLBenchSDFBarrier", "VoxelizedSDFCache",
        "make_box_mesh", "make_sphere_mesh",
    ]
