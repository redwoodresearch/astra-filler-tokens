"""API keys from environment variables: OPENAI_API_KEY, OPENROUTER_API_KEY, ANTHROPIC_API_KEY (and HF_TOKEN for the
gated HLE dataset). Keys are only held in-process and never printed."""

import os


def secret_field(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"set {name} in the environment")
    return val


def openai_key() -> str:
    return secret_field("OPENAI_API_KEY")
