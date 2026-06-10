"""llm-finetuning: end-to-end LLM lifecycle toolkit (UFPI - Tópicos em IA).

Importing the package registers the built-in components (model providers,
dataset loaders, metrics, evaluators) into their registries. Third-party imports
(torch, transformers, pypdf) are deferred to call time.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Populate the registries as a side effect of importing the package.
from . import data as _data  # noqa: F401
from . import evaluation as _evaluation  # noqa: F401
from . import models as _models  # noqa: F401

__all__ = ["__version__"]
