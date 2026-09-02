"""Robust cross-modal tactile representation learning modules."""

from .config import CrossModalTactileNetworkCfg
from .models.network import RobustCrossModalTactileNetwork
from .models.latent_alignment import (
    AffineTactileLatentAlignmentNetwork,
    ProjectionHead,
    TactileLatentAlignmentNetwork,
)
from .models.tri_modal_cross_autoencoder import TriModalCrossAutoencoder
from .tri_modal_config import TriModalCrossAutoencoderCfg

__all__ = [
    "CrossModalTactileNetworkCfg",
    "RobustCrossModalTactileNetwork",
    "TriModalCrossAutoencoder",
    "TriModalCrossAutoencoderCfg",
    "ProjectionHead",
    "TactileLatentAlignmentNetwork",
    "AffineTactileLatentAlignmentNetwork",
]
