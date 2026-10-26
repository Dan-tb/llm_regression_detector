"""Load versioned prompt YAML files from /prompts."""
from __future__ import annotations

from pathlib import Path

import yaml

from .models import PromptConfig


def load_prompt(path: str | Path) -> PromptConfig:
    with open(path, encoding="utf-8") as f:
        return PromptConfig(**yaml.safe_load(f))


def changed_prompt_files(directory: str | Path) -> list[Path]:
    return sorted(Path(directory).glob("*.yaml"))
