import json
import os
from typing import Any, Dict, Optional, Tuple

DEFAULT_PROFILES_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "configs", "model_profiles.json")
)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def apply_model_profile(cfg: Dict[str, Any], profiles_path: Optional[str] = None) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Apply MODEL_KEY-based profile settings on top of cfg.

    Precedence order:
      1) cfg base
      2) profile (overrides cfg for overlapping keys)
      3) cfg[MODEL_PROFILE_OVERRIDES] (overrides profile)
    """
    model_key = cfg.get("MODEL_KEY")
    if not model_key:
        return cfg, None

    profiles_path = profiles_path or cfg.get("MODEL_PROFILES_PATH") or DEFAULT_PROFILES_PATH
    if not os.path.exists(profiles_path):
        raise FileNotFoundError(f"MODEL_PROFILES_PATH not found: {profiles_path}")

    profiles = _load_json(profiles_path)
    if model_key not in profiles:
        raise KeyError(f"MODEL_KEY '{model_key}' not found in {profiles_path}")

    profile = profiles.get(model_key, {}) or {}
    merged = dict(cfg)
    merged.update(profile)

    overrides = cfg.get("MODEL_PROFILE_OVERRIDES")
    if isinstance(overrides, dict):
        merged.update(overrides)

    return merged, model_key


def load_config_with_profile(config_path: str) -> Dict[str, Any]:
    cfg = _load_json(config_path)
    merged, _ = apply_model_profile(cfg)
    return merged
