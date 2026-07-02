"""MOSS-TTS-ComfyUI — OpenMOSS MOSS-TTS-Local-Transformer-v1.5 voice cloning."""

from __future__ import annotations

import logging

logger = logging.getLogger("MOSS-TTS-ComfyUI")

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    logger.info(f"[MOSS-TTS] registered {len(NODE_CLASS_MAPPINGS)} node(s).")
except Exception as exc:  # pragma: no cover
    logger.exception("[MOSS-TTS] failed to register nodes: %s", exc)
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
