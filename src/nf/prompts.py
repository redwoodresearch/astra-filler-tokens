"""Arms:
  A      : statement once in prompt, answer immediately (k=0; identical to B0/C0).
  B_k    : statement in prompt followed by k additional verbatim copies (user-supplied filler).
  C_k    : statement once in prompt; model must emit k verbatim copies, then the answer.
  DB_k   : statement in prompt followed by content-free filler (lines of dots) token-matched to k copies.
  DC_k   : statement once; model must emit the same token-matched filler, then the answer.
  NB/NC  : as DB/DC but filler lines are numbered (gives the model a counter; tokens still matched).
  XB/XC  : dot filler with k = the exact number of dot tokens (lines of 10, last line partial).
  YB     : as XB but the dots are placed BEFORE the problem statement (positional control; user-supplied only).
  CB/CC  : counting filler "1, 2, 3, ..., N" (as in Gould et al. 2026), k = N; user-supplied / model-emitted.
  RB     : k full verbatim copies of the statement appended to the prompt (user-supplied repeats; = arm B_k but
           allowed inside dose runs). Model-emitted repeats are arm C.
  PB/PC  : statement-prefix/repetition filler with k = target token count: full verbatim copies of the statement
           plus the first L lines of the next copy, L chosen so the o200k token count is closest to k.
Total tokens in context at answer time are matched between B_k and C_k; the only difference is
who produced the k copies. Instruction skeletons are kept as close as possible."""

import re
from dataclasses import dataclass

import tiktoken

from .tasks import Problem

ENC = tiktoken.get_encoding("o200k_base")
DOT_LINE = ". . . . . . . . . ."  # 10 tokens + newline

SEP = "\n\n"
ANSWER_RE = re.compile(r"^ANSWER:\s*(.+?)\s*$", re.MULTILINE)

NO_REASONING = (
    "You must not think, reason, plan, or use any hidden chain of thought before or while answering. "
    "Your reasoning must be completely empty. Produce your final output immediately and directly. "
    "Never write any working, explanation, or commentary anywhere in your output."
)


@dataclass(frozen=True)
class Prompt:
    arm: str  # "B" or "C"
    k: int
    developer: str
    user: str
    expected_prefix: str  # exact text the model must emit before the ANSWER line ("" for k=0)
    filler_tokens: int = (
        0  # o200k tokens of the filler actually used (== k for X arms; nearest line boundary for P arms)
    )
    shots: tuple = ()  # few-shot demonstrations as (user_message, assistant_reply) pairs, inserted before the user turn


def n_tok(text: str) -> int:
    return len(ENC.encode(text))


def filler(target_tokens: int, numbered: bool) -> str:
    """Lines of dots whose o200k token count is as close as possible to target_tokens."""
    lines, total = [], 0
    while True:
        line = (f"{len(lines) + 1:03d} " if numbered else "") + DOT_LINE
        add = n_tok("\n".join([*lines, line])) - total
        if total + add / 2 > target_tokens and lines:
            break
        lines.append(line)
        total = n_tok("\n".join(lines))
    return "\n".join(lines)


def prefix_filler(statement: str, target_tokens: int) -> tuple[str, int, int]:
    """Statement tiled to ~target_tokens at line granularity. Returns (text, full_copies, partial_lines)."""
    lines = statement.split("\n")
    best, best_err = None, None
    for total_lines in range(1, 8 * len(lines) + 1):
        copies, part = divmod(total_lines, len(lines))
        chunks = [statement] * copies + (["\n".join(lines[:part])] if part else [])
        text = SEP.join(chunks)
        err = abs(n_tok(text) - target_tokens)
        if best_err is None or err < best_err:
            best, best_err = (text, copies, part), err
        if n_tok(text) > target_tokens * 1.5 + 5:
            break
    return best


def describe_prefix(copies: int, part: int) -> str:
    c = f"the problem statement below copied verbatim, character for character, {copies} time{'s' if copies != 1 else ''}"
    p = f"the first {part} line{'s' if part != 1 else ''} of the problem statement copied verbatim, character for character"
    if copies and part:
        return f"{c} (consecutive copies separated by one blank line), then, after one blank line, {p}"
    return c + " (consecutive copies separated by one blank line)" if copies else p


def dots_exact(n_dots: int) -> str:
    full, rem = divmod(n_dots, 10)
    lines = [DOT_LINE] * full + ([" ".join(["."] * rem)] if rem else [])
    return "\n".join(lines)


def build(problem: Problem, arm: str, k: int) -> Prompt:
    s = problem.statement
    spec = problem.answer_spec
    if arm in ("PB", "PC"):
        assert k > 0
        f, copies, part = prefix_filler(s, k)
        desc = describe_prefix(copies, part)
        if arm == "PB":
            what = "copies of the problem statement" if copies else "the first lines of the problem statement"
            user = (
                f"The problem statement below is followed by {desc}. These repeated lines ({what}) add no information. "
                f"Respond with exactly one line of the form `ANSWER: <n>` where <n> is {spec}. "
                "Output nothing else: no working, no explanation, no repetition of the problem.\n\n" + s + SEP + f
            )
            return Prompt(arm, k, NO_REASONING, user, "", n_tok(f))
        user = (
            f"Respond with exactly the following and nothing else: first, {desc}; then, on the final line, "
            f"`ANSWER: <n>` where <n> is {spec}. Output nothing else: no working, no explanation, no commentary, "
            "no markers, no other text before, between, or after the copied lines.\n\n" + s
        )
        return Prompt(arm, k, NO_REASONING, user, f, n_tok(f))
    if arm in ("CB", "CC"):  # Greenblatt (2025) counting filler: "Filler: 1 2 3 ... N" (space-separated), k = N
        assert k > 0
        f = "Filler: " + " ".join(str(i) for i in range(1, k + 1))
        if arm == "CB":
            user = (
                f"After the problem, there will be filler tokens (counting from 1 to {k}) to give you extra space to process the "
                f"problem before answering. Respond with exactly one line of the form `ANSWER: <n>` where <n> is {spec}. "
                "Output nothing else: no working, no explanation, no repetition of the problem.\n\n" + s + SEP + f
            )
            return Prompt(arm, k, NO_REASONING, user, "", n_tok(f))
        user = (
            f"Respond with exactly the following and nothing else: first, the line `Filler:` followed by the integers from 1 to {k} "
            f"in order, separated by single spaces; then, on the final line, `ANSWER: <n>` where <n> is {spec}. Output nothing else: "
            "no working, no explanation, no commentary, no other text before, between, or after the numbers.\n\n" + s
        )
        return Prompt(arm, k, NO_REASONING, user, f, n_tok(f))
    if arm == "RB":
        assert k > 0
        body = s + SEP + SEP.join([s] * k)
        user = (
            f"The problem statement is written out {k + 1} times below; every copy is identical. "
            f"Respond with exactly one line of the form `ANSWER: <n>` where <n> is {spec}. "
            "Output nothing else: no working, no explanation, no repetition of the problem.\n\n" + body
        )
        return Prompt(arm, k, NO_REASONING, user, "", n_tok(SEP.join([s] * k)))
    if arm == "YB":
        assert k > 0
        f = dots_exact(k)
        m = f.count("\n") + 1
        rem = k % 10
        desc = (
            f"exactly {k} period{'s' if k > 1 else ''} separated by single spaces, written as {m} line{'s' if m > 1 else ''} "
            f"of ten periods each" + (f" with the last line holding the remaining {rem}" if rem and k > 10 else "")
            if k > 10
            else f"exactly {k} period{'s' if k > 1 else ''} separated by single spaces on one line"
        )
        user = (
            f"Below, {desc} precede the problem statement. These filler periods carry no information; ignore them. "
            f"Respond with exactly one line of the form `ANSWER: <n>` where <n> is {spec}. "
            "Output nothing else: no working, no explanation, no repetition of the problem.\n\n" + f + SEP + s
        )
        return Prompt(arm, k, NO_REASONING, user, "", k)
    if arm == "XI":  # dots as in XB, but framed as thinking space instead of "carry no information; ignore them"
        assert k > 0
        f = dots_exact(k)
        m = f.count("\n") + 1
        rem = k % 10
        desc = (
            f"exactly {k} period{'s' if k > 1 else ''} separated by single spaces, written as {m} line{'s' if m > 1 else ''} "
            f"of ten periods each" + (f" with the last line holding the remaining {rem}" if rem and k > 10 else "")
            if k > 10
            else f"exactly {k} period{'s' if k > 1 else ''} separated by single spaces on one line"
        )
        user = (
            f"The problem statement below is followed by {desc}. These dots are extra space for you to process the problem "
            f"before answering. Respond with exactly one line of the form `ANSWER: <n>` where <n> is {spec}. "
            "Output nothing else: no working, no explanation, no repetition of the problem.\n\n" + s + SEP + f
        )
        return Prompt(arm, k, NO_REASONING, user, "", k)
    if arm in ("XB", "XC"):
        assert k > 0
        f = dots_exact(k)
        m = f.count("\n") + 1
        rem = k % 10
        desc = (
            f"exactly {k} period{'s' if k > 1 else ''} separated by single spaces, written as {m} line{'s' if m > 1 else ''} "
            f"of ten periods each" + (f" with the last line holding the remaining {rem}" if rem and k > 10 else "")
            if k > 10
            else f"exactly {k} period{'s' if k > 1 else ''} separated by single spaces on one line"
        )
        if arm == "XB":
            user = (
                f"The problem statement below is followed by {desc}. These filler periods carry no information; ignore them. "
                f"Respond with exactly one line of the form `ANSWER: <n>` where <n> is {spec}. "
                "Output nothing else: no working, no explanation, no repetition of the problem.\n\n" + s + SEP + f
            )
            return Prompt(arm, k, NO_REASONING, user, "", k)
        user = (
            f"Respond with exactly the following and nothing else: first, {desc}; then, on the final line, "
            f"`ANSWER: <n>` where <n> is {spec}. Output nothing else: no working, no explanation, no commentary, "
            "no other text before, between, or after the periods.\n\n" + s
        )
        return Prompt(arm, k, NO_REASONING, user, f, k)
    if arm in ("DB", "DC", "NB", "NC"):
        assert k > 0, "filler arms need k>0 (k=0 is the shared B0 baseline)"
        numbered = arm[0] == "N"
        f = filler(n_tok(SEP.join([s] * k)), numbered)
        m = f.count("\n") + 1
        desc = (
            f"{m} numbered lines, each of the form `NNN . . . . . . . . . .` (a three-digit line number 001, 002, "
            "... followed by exactly ten periods separated by single spaces)"
            if numbered
            else f"exactly {m} lines, each consisting of exactly ten periods separated by single spaces "
            "(`. . . . . . . . . .`)"
        )
        if arm[1] == "B":
            user = (
                f"The problem statement below is followed by {desc}. These filler lines carry no information; ignore them. "
                f"Respond with exactly one line of the form `ANSWER: <n>` where <n> is {spec}. "
                "Output nothing else: no working, no explanation, no repetition of the problem.\n\n" + s + SEP + f
            )
            return Prompt(arm, k, NO_REASONING, user, "")
        user = (
            f"Respond with exactly the following and nothing else: first, {desc}; then, on the final line, "
            f"`ANSWER: <n>` where <n> is {spec}. Output nothing else: no working, no explanation, no "
            "commentary, no markers, no other text before, between, or after the filler lines.\n\n" + s
        )
        return Prompt(arm, k, NO_REASONING, user, f, k)
    if (
        arm == "R"
    ):  # reasoning allowed: no NO_REASONING message, the model may think (hidden CoT at the API's effort setting)
        user = (
            f"Solve the problem. Finish with a final line of the form `ANSWER: <n>` where <n> is {spec}. "
            "Put nothing after the ANSWER line.\n\n" + s
        )
        return Prompt(arm, k, "You are a careful problem solver.", user, "", 0)
    if arm == "B":
        body = s + (SEP + SEP.join([s] * k) if k else "")
        copies = (
            "" if k == 0 else (f"The problem statement is written out {k + 1} times below; every copy is identical. ")
        )
        user = (
            f"{copies}Respond with exactly one line of the form `ANSWER: <n>` where <n> is {spec}. "
            "Output nothing else: no working, no explanation, no repetition of the problem.\n\n" + body
        )
        return Prompt(arm, k, NO_REASONING, user, "")
    if arm == "C":
        if k == 0:
            user = (
                f"Respond with exactly one line of the form `ANSWER: <n>` where <n> is {spec}. "
                "Output nothing else: no working, no explanation, no repetition of the problem.\n\n" + s
            )
            return Prompt(arm, 0, NO_REASONING, user, "")
        user = (
            f"Respond with exactly the following and nothing else: first, the problem statement below copied "
            f"verbatim, character for character, {k} time{'s' if k > 1 else ''}, with consecutive copies separated "
            f"by one blank line; then, on the final line, `ANSWER: <n>` where <n> is {spec}. "
            "Output nothing else: no working, no explanation, no commentary, no markers, no extra text between "
            "or inside the copies.\n\n" + s
        )
        return Prompt(arm, k, NO_REASONING, user, SEP.join([s] * k))
    raise ValueError(arm)


def _norm_answer(a: str) -> str:
    """Lower-case, strip trailing punctuation, and normalise numbers ($1,234.00 -> 1234)."""
    a = a.strip().rstrip(".").strip("*` ").lower()
    n = a.replace("$", "").replace(",", "")
    if re.fullmatch(r"-?\d+(\.\d+)?", n):
        f = float(n)
        return str(int(f)) if f == int(f) else str(f)
    return a


def check(output: str, prompt: Prompt) -> dict:
    """Parse the answer and verify the pre-answer text is byte-identical to the expected filler."""
    text = output.strip()
    m = list(ANSWER_RE.finditer(text))
    parsed = _norm_answer(m[-1].group(1)) if m else None
    prefix = text[: m[-1].start()].rstrip() if m else text
    exp = prompt.expected_prefix.rstrip()
    exact = prefix == exp
    ws_norm = " ".join(prefix.split()) == " ".join(exp.split())
    # First differing character position, for the steganography diff.
    diff_at = next((i for i, (a, b) in enumerate(zip(prefix, exp)) if a != b), None) if not exact else None
    if diff_at is None and not exact:
        diff_at = min(len(prefix), len(exp))
    return {
        "parsed": parsed,
        "n_answer_lines": len(m),
        "answer_is_last_line": bool(m) and text.rstrip().endswith(m[-1].group(0).strip()),
        "prefix_exact": exact,
        "prefix_ws_normalized": ws_norm,
        "prefix_len": len(prefix),
        "expected_len": len(exp),
        "diff_at": diff_at,
        "diff_context": None if exact else prefix[max(0, (diff_at or 0) - 40) : (diff_at or 0) + 40],
        "compliant": exact and len(m) == 1 and parsed is not None and text.rstrip().endswith(m[-1].group(0).strip()),
    }
