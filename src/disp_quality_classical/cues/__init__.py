from .validity import compute_validity_cue
from .smoothness import compute_smoothness_cue
from .multiscale import compute_multiscale_cue
from .edge_coherence import compute_edge_coherence_cue
from .depth_sensitivity import compute_depth_sensitivity_cue
from .hole_clustering import compute_hole_clustering_cue
from .local_outlier import compute_local_outlier_cue

__all__ = [name for name in globals() if name.startswith("compute_")]
