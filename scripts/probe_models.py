"""List gpt-6 / gpt-5.6 model ids visible to a key (free endpoint; prints no secrets).
usage: uv run scripts_probe_models.py [ENV_VAR_NAME]"""

import sys

from openai import OpenAI

from nf.keys import openai_key

name = sys.argv[1] if len(sys.argv) > 1 else None
c = OpenAI(api_key=openai_key(name))
ids = sorted(m.id for m in c.models.list())
print(f"{len(ids)} models;", [i for i in ids if i.startswith(("gpt-6", "gpt-5.6", "gpt-5.5"))])
