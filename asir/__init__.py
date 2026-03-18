"""
ASIR — Acoustic Semantic IR for Hearing Aids
7-layer semantic instruction set architecture using DSPy + GEPA.
"""

__version__ = "0.4.0"

from asir.harness import AcousticSemanticHarness
from asir.types import (
    RawSignal, DSPParameterSet, AcousticFeatures,
    PerceptualDescription, SceneUnderstanding, ProcessingStrategy,
    UserIntent, UserPreferences,
)
