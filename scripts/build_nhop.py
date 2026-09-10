"""Compose the N-hop natural-facts questions (data/nhop_{2..7}.jsonl) from rgreenblatt/multi_hop's fact tables.

Requires a checkout of https://github.com/rgreenblatt/multi_hop (pass its path as the first argument). Procedure:
start from every base problem (NUMBER_GENERATORS x ALL_CONSUMERS via gen_hop_generic); then, six times, turn every
numeric-answer problem with fewer than 7 hops into a new number generator whose single item is (answer, "(question)",
chain) and compose it with every consumer, keeping only chains whose fact tables (mapping_ids) are all distinct; finally,
for each hop count 2..7, shuffle with seed 7, drop duplicates by (answer, first fact) and keep the first 150.
Usage: uv run scripts/build_nhop.py /path/to/multi_hop [--out data]"""

import importlib.util
import json
import os
import random
import sys
from pathlib import Path


def load_generator(repo: Path):
    cwd = os.getcwd()
    os.chdir(repo)  # the constants module opens data/*.json relative to cwd
    try:
        sys.path.insert(0, str(repo))
        spec = importlib.util.spec_from_file_location("generate_dataset", repo / "generate_dataset.py")
        gd = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gd)
        return gd
    finally:
        os.chdir(cwd)


def compose(gd, seed: int = 128):
    def all_problems(generators):
        out = []
        for gen in generators:
            for cons in gd.ALL_CONSUMERS:
                out.extend(gd.gen_hop_generic(gen, cons, seed=seed, num=None))
        return out

    frontier = all_problems(gd.NUMBER_GENERATORS)
    pool = list(frontier)
    for _ in range(6):
        gens = []
        for p in frontier:
            if isinstance(p["answer"], int) and len(p["chain"]) < 7:
                gens.append({"id": p["type"], "items": [(p["answer"], f"({p['question'].rstrip('?')})", p["chain"])]})
        new = [p for p in all_problems(gens) if len({h["mapping_id"] for h in p["chain"]}) == len(p["chain"])]
        pool.extend(new)
        frontier = new
    return pool


def main():
    repo = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "--out" else "data")
    gd = load_generator(repo)
    pool = compose(gd)
    spec = "the answer only (a name, a US state, an element, a motto/flower, or a number)"
    for hops in range(2, 8):
        rows = [p for p in pool if len(p["chain"]) == hops]
        random.Random(7).shuffle(rows)
        seen, keep = set(), []
        for p in rows:
            key = (str(p["answer"]), p["chain"][0]["fact"])
            if key in seen:
                continue
            seen.add(key)
            keep.append(
                {
                    "id": f"nhop{hops}_{len(keep)}",
                    "source": "rgreenblatt/multi_hop (composed)",
                    "statement": p["question"],
                    "answer": str(p["answer"]),
                    "chain": p["chain"],
                    "type": p["type"],
                    "answer_spec": spec,
                    "chance": 0.0,
                }
            )
            if len(keep) == 150:
                break
        (out / f"nhop_{hops}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in keep))
        print(hops, "hops:", len(rows), "candidates ->", len(keep), "kept")


if __name__ == "__main__":
    main()
