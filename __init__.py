"""MOSS-TTS-ComfyUI — OpenMOSS MOSS-TTS v1.5 (1.7B Local-Transformer + 8B full model).

TTS, zero-shot voice cloning, audio continuation and duration control.

NODE_CLASS_MAPPINGS is re-exported at module top level as a plain import so the
ComfyUI Registry's static (AST) node parser can discover all five nodes. Do NOT
wrap this import in a try/except with an empty-dict fallback — the parser then
records the empty dict and the pack shows "No nodes found". ComfyUI's own custom-
node loader already catches and logs import errors gracefully, so no guard here.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
