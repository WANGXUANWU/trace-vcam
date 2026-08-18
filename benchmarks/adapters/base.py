"""Abstract fit/predict/factor interface for benchmark implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from ..data import SubjectDataset


class AdapterError(RuntimeError):
    """Base error carrying a stable audit code."""

    code = "adapter_error"


class AdapterUnavailable(AdapterError):
    code = "adapter_unavailable"


class UnsupportedPrediction(AdapterError):
    code = "prediction_not_supported"


@dataclass(frozen=True)
class PreflightReport:
    ready: bool
    version: str
    code: str = "ready"
    message: str = ""
    environment: Mapping[str, object] = field(default_factory=dict)


@dataclass
class FitArtifact:
    """In-memory fit plus serializable audit information."""

    model: Any
    method: str
    version: str
    tuning: dict[str, object]
    converged: bool
    selected_blocks: tuple[int, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)


class BenchmarkAdapter(ABC):
    """All literature methods enter the study through this interface."""

    label: str

    @abstractmethod
    def preflight(self) -> PreflightReport:
        """Check availability without modifying the environment."""

    @abstractmethod
    def fit(
        self,
        train: SubjectDataset,
        *,
        seed: int,
        tuning: Mapping[str, object],
    ) -> FitArtifact:
        """Fit using training subjects only."""

    @abstractmethod
    def predict(self, artifact: FitArtifact, test: SubjectDataset) -> NDArray[np.float64]:
        """Predict rows belonging exclusively to held-out subjects."""

    def factor_curves(self, artifact: FitArtifact) -> tuple[dict[str, object], ...]:
        """Return serialized identified factor curves when the method exposes them."""

        return ()
