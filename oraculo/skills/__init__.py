"""Biblioteca de skills: importar este pacote registra as 32 skills."""
from . import control, intelligence, macro, market, predictive  # noqa: F401
from .base import REGISTRY, SkillContext, SkillResult, SkillStatus, run_skill  # noqa: F401
