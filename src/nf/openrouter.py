"""OpenRouter model specs and endpoint pins.

Spec syntax: "or/<author>/<slug>@<effort>" routes via OpenRouter; "<openai-id>:<effort>" goes to the OpenAI API.
effort "off" sends reasoning.enabled=false; "none"/"low"/... send reasoning.effort; no effort sends nothing.
PINS: per-model `provider` dict. Open-weight models are pinned to one high-precision endpoint with
allow_fallbacks=False (reproducibility-first); proprietary models are single-served so only data policy is set.
The served provider and generation id are stored on every record."""

import json
from pathlib import Path

PROPRIETARY = {"data_collection": "deny", "allow_fallbacks": False}


def parse_spec(spec: str) -> tuple[str, str, str | None]:
    if spec.startswith(
        "ant/"
    ):  # Anthropic API: "ant/<model>@off" (thinking disabled) or "@<effort>" (thinking on, effort set)
        model, _, effort = spec[4:].partition("@")
        return "anthropic", model, (effort or None)
    if spec.startswith("or/"):
        body = spec[3:]
        model, _, effort = body.partition("@")
        return "openrouter", model, (effort or None)
    model, _, effort = spec.rpartition(":") if ":" in spec else (spec, "", "")
    return "openai", model, (None if effort in ("", "default") else effort)


_PINS_FILE = Path(__file__).resolve().parents[2] / "data" / "openrouter_pins.json"
PINS: dict[str, dict] = json.loads(_PINS_FILE.read_text()) if _PINS_FILE.exists() else {}
