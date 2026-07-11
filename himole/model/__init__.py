"""Model components for baseline LoRA and the strict HiMoLE scaffold."""

from himole.model.experts import KnowledgeCompetitionGroup, LoRAExpert
from himole.model.layer import HiMoLEFFNLayer
from himole.model.patching import attach_himole_to_model, freeze_base_model
from himole.model.routing import HierarchicalRouter, RoutingOutput

__all__ = [
    "HiMoLEFFNLayer",
    "HierarchicalRouter",
    "KnowledgeCompetitionGroup",
    "LoRAExpert",
    "RoutingOutput",
    "attach_himole_to_model",
    "freeze_base_model",
]
