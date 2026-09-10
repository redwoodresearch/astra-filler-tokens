"""Answer grading for the natural-fact n-hop task, ported from rgreenblatt/multi_hop eval_multi_hop.py
(normalize_answer + check_answer + remove_middle_names). Mottos/flowers are vendored in data/us_state_mottos_flowers.json."""

import json
import re
from pathlib import Path

from unidecode import unidecode

_TABLES = json.loads((Path(__file__).resolve().parents[2] / "data" / "us_state_mottos_flowers.json").read_text())


class _C:  # state mottos / flowers (from rgreenblatt/multi_hop generate_dataset_constants.py), used to skip name normalisation
    US_STATE_MOTTOS = _TABLES["US_STATE_MOTTOS"]
    US_STATE_FLOWERS = _TABLES["US_STATE_FLOWERS"]


def remove_middle_names(name_str: str) -> str:
    """
    Remove middle names from a person's name (conservative approach).
    Only applied to names with 3+ words where all words are alphabetic.
    Preserves suffixes like Jr., Sr., I, II, III.
    """
    words = name_str.split()
    if len(words) < 3:
        return name_str
    if len(words) > 4:
        return name_str

    # Only apply to names that look like person names (all words are alphabetic)
    # Allow apostrophes and hyphens which are common in names
    if not all(word.replace("'", "").replace("-", "").isalpha() for word in words):
        return name_str

    # Preserve common suffixes
    suffixes = {"jr", "sr", "i", "ii", "iii", "iv", "v"}
    if words[-1] in suffixes and len(words) in [3, 4]:
        if len(words) == 4:
            return f"{words[0]} {words[-2]} {words[-1]}"
        else:
            # Only 3 words with suffix (e.g., "John Smith Jr"), don't modify
            return name_str
    if len(words) > 3:
        return name_str

    if len(words) == 3:
        return f"{words[0]} {words[2]}"

    raise ValueError("shoudn't be reachable")


def normalize_answer(answer_str: str, skip_middle_name_normalization: bool = False) -> str:
    """Normalize answer string for comparison."""
    answer_str = answer_str.strip().lower()
    answer_str = unidecode(answer_str)

    # Remove common prefixes
    for prefix in ["answer:", "the answer is", "it is", "it's"]:
        if answer_str.startswith(prefix):
            answer_str = answer_str[len(prefix) :].strip()

    answer_str = answer_str.replace("t.s.", "t. s.")
    answer_str = answer_str.replace("j.j.", "j. j.")
    answer_str = answer_str.removeprefix("mr. ").strip()
    answer_str = answer_str.removeprefix("sir. ").strip()
    answer_str = answer_str.removeprefix("mr ").strip()
    answer_str = answer_str.removeprefix("sir ").strip()
    answer_str = answer_str.removeprefix("lord ").strip()

    # Remove parenthetical explanations (e.g., "Cardinal (Northern Cardinal)" -> "Cardinal")
    answer_str = re.sub(r"\s*\([^)]*\)", "", answer_str).strip()

    answer_str = answer_str.replace("robert bruce merrifield", "bruce merrifield").strip()
    answer_str = answer_str.replace("oscar arias sanchez", "oscar arias").strip()
    answer_str = answer_str.replace("john boyd orr", "boyd orr").strip()
    answer_str = answer_str.replace("hermann emil fischer", "emil fischer").strip()
    answer_str = answer_str.replace("petrus debye", "peter debye").strip()
    answer_str = answer_str.replace("adolf otto reinhold windaus", "adolf windaus").strip()
    answer_str = answer_str.replace("william randal cremer", "randal cremer").strip()
    answer_str = answer_str.replace("rigoberta menchu tum", "rigoberta menchu").strip()

    answer_str = answer_str.replace("fatti maschi,", "fatti maschii,").strip()

    answer_str = answer_str.replace("washington, d.c.", "district of columbia").strip()
    answer_str = answer_str.replace("d.c.", "district of columbia").strip()

    answer_str = answer_str.replace("audemus jura nostra defendere", "we dare defend our rights").strip()
    answer_str = answer_str.replace("and ", "").strip()

    answer_str = answer_str.replace("-", "").strip()
    answer_str = answer_str.replace(",", "").strip()
    answer_str = answer_str.replace(".", "").strip()
    answer_str = answer_str.replace("'", "").strip()  # Remove apostrophes
    answer_str = answer_str.replace("`", "").strip()  # Remove backticks (Hawaiian ʻokina after unidecode)

    # some overkill here
    answer_str = answer_str.strip(".,!?;:").strip()
    answer_str = answer_str.removeprefix("the ").strip()
    answer_str = answer_str.removeprefix("[").removesuffix("]").strip()
    answer_str = answer_str.strip(".,!?;:").strip()
    answer_str = answer_str.removeprefix("the ").strip()

    answer_str = answer_str.removeprefix("northern").strip()
    answer_str = answer_str.removeprefix("western").strip()
    answer_str = answer_str.removeprefix("eastern").strip()
    answer_str = answer_str.removeprefix("southern").strip()
    answer_str = answer_str.removeprefix("american").strip()
    answer_str = answer_str.replace("hawaiian hibiscus", "hibiscus")
    answer_str = answer_str.replace("white hawthorn blossom", "hawthorn")
    answer_str = answer_str.replace("common meadow violet", "violet")
    answer_str = answer_str.replace("yucca flower", "yucca")

    if (not skip_middle_name_normalization) and (answer_str not in NORMALIZED_NON_NAME_SET):
        # Remove middle names from person names (conservative, only affects 3+ word names)
        answer_str = remove_middle_names(answer_str)

    return answer_str


NORMALIZED_NON_NAME_SET = {
    normalize_answer(x, skip_middle_name_normalization=True)
    for x in [*_C.US_STATE_MOTTOS.values(), *_C.US_STATE_FLOWERS.values()]
}


def check_answer(predicted: str, correct) -> bool:
    """Check if predicted answer matches correct answer."""
    pred_norm = normalize_answer(str(predicted))
    correct_norm = normalize_answer(str(correct))

    # Direct match (includes middle name removal since it's in normalize_answer)
    if pred_norm == correct_norm:
        return True

    # For numeric answers, try parsing
    try:
        pred_num = int(pred_norm)
        correct_num = int(correct)
        if pred_num == correct_num:
            return True
    except ValueError:
        pass

    return False
