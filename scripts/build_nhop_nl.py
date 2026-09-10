"""Render the n-hop chains (data/nhop_{k}.jsonl) as nested English in the style of nocot-bench's realhop_nl:
one noun phrase per hop wrapped around the previous one, seed entity last, no parenthesised arithmetic. Same ids, chains and
answers; writes data/nhopnl_{k}.jsonl."""

import json
import re
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"

OSCAR = {
    "best_actor": "Best Actor",
    "best_actress": "Best Actress",
    "best_supporting_actor": "Best Supporting Actor",
    "best_supporting_actress": "Best Supporting Actress",
}
NOBEL = {
    "physics": "the Nobel Prize in Physics",
    "chemistry": "the Nobel Prize in Chemistry",
    "literature": "the Nobel Prize in Literature",
    "peace": "the Nobel Peace Prize",
}


def seed_phrase(fact: str) -> str:
    """Seed hops are stored as questions with numeric answers; turn them into noun phrases."""
    f = fact.strip().rstrip("?")
    m = re.match(r"At what age did (.+) die", f)
    if m:
        return f"the age at which {m.group(1)} died"
    m = re.match(r"Atomic number of (.+)", f)
    if m:
        return f"the atomic number of {m.group(1)}"
    m = re.match(r"How many representatives does (.+) have in the US House of Representatives", f)
    if m:
        return f"the number of seats {m.group(1)} has in the US House of Representatives"
    m = re.match(r"How many seats are in the lower house of (.+)'s state legislature", f)
    if m:
        return f"the number of seats in the lower house of {m.group(1)}'s state legislature"
    m = re.match(r"How many (.+)", f)
    if m:
        return f"the number of {m.group(1)}"
    m = re.match(r"What is the number of (.+)", f)
    if m:
        return f"the number of {m.group(1)}"
    m = re.match(r"What is (.+)", f)
    if m:
        return m.group(1)
    m = re.match(r"(\d+)(?:st|nd|rd|th) Oscar (.+) winner", f)  # order_to_oscar as a seed
    if m:
        return f"the {m.group(2)} winner at the {m.group(1)}{'th' if not m.group(1).endswith(('1', '2', '3')) or m.group(1).endswith(('11', '12', '13')) else {'1': 'st', '2': 'nd', '3': 'rd'}[m.group(1)[-1]]} Academy Awards"
    m = re.match(r"(Nobel .+) in (\d{4})", f)
    if m:
        return f"the winner of the {m.group(1)} in {m.group(2)}"
    m = re.match(r"Oscar (.+) in (\d{4})", f)
    if m:
        return f"the {m.group(1)} winner for a film released in {m.group(2)}"
    m = re.match(r"State that joined the union (\d+)(?:st|nd|rd|th)", f)
    if m:
        return f"the US state that was the {m.group(1)}th to join the union"
    raise ValueError(fact)


def year_from(prev: str) -> str:
    return f"the 20th-century year whose last two digits equal {prev}"


def wrap(mapping_id: str, prev: str) -> str:
    """Noun phrase for the value produced by mapping_id applied to the entity/number described by prev."""
    if mapping_id == "state_order_to_state":
        return f"the US state whose position in the order of joining the union equals {prev}"
    if mapping_id == "atomic_number_to_element":
        return f"the chemical element whose atomic number equals {prev}"
    if mapping_id == "element_to_atomic_number":
        return f"the atomic number of {prev}"
    if mapping_id == "state_to_counties":
        return f"the number of county-equivalents in {prev}"
    if mapping_id == "state_to_flower":
        return f"the state flower of {prev}"
    if mapping_id == "state_to_motto":
        return f"the state motto of {prev}"
    if mapping_id == "year_to_miss_america":
        return f"the Miss America winner for {year_from(prev)}"
    m = re.match(r"order_to_oscar_(.+)", mapping_id)
    if m:
        return f"the {OSCAR[m.group(1)]} winner at the Academy Awards ceremony whose number equals {prev}"
    m = re.match(r"year_to_oscar_(.+)", mapping_id)
    if m:
        return f"the {OSCAR[m.group(1)]} winner for a film released in {year_from(prev)}"
    m = re.match(r"year_to_nobel_(.+)", mapping_id)
    if m:
        return f"the winner of {NOBEL[m.group(1)]} in {year_from(prev)}"
    if mapping_id.endswith(
        "_to_birth_day"
    ):  # head-first so the sentence stays right-branching (no stacked "was born was born")
        return f"the day-of-month of the birth of {prev}"
    if mapping_id.endswith("_to_birth_year"):
        return f"the birth year of {prev}"
    raise ValueError(mapping_id)


def question(phrase: str, last_mapping: str) -> str:
    if last_mapping.endswith("_to_birth_year"):
        return f"In what year was {phrase[len('the birth year of ') :]} born?"
    if last_mapping.endswith("_to_birth_day"):
        return f"On what day of the month was {phrase[len('the day-of-month of the birth of ') :]} born?"
    if last_mapping in ("state_to_counties", "element_to_atomic_number"):
        return f"What is {phrase}?"
    if last_mapping in ("state_to_flower", "state_to_motto"):
        return f"What is {phrase}?"
    if last_mapping in ("state_order_to_state", "atomic_number_to_element"):
        return f"What is {phrase}?"
    return f"Who is {phrase}?"


def render(chain: list[dict]) -> str:
    phrase = seed_phrase(chain[0]["fact"])
    for hop in chain[1:]:
        phrase = wrap(hop["mapping_id"], phrase)
    return question(phrase, chain[-1]["mapping_id"])


if __name__ == "__main__":
    for k in range(2, 8):
        rows = [json.loads(line) for line in (DATA / f"nhop_{k}.jsonl").open()]
        for r in rows:
            r["statement_original"] = r["statement"]
            r["statement"] = render(r["chain"])
        (DATA / f"nhopnl_{k}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(k, "hops:", rows[0]["statement"])
