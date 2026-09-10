"""Programmatic graders for HLE and LiveBench answer-format tasks.

HLE's official metric uses an LLM judge; here a strict normalized match is the primary metric (an optional judge pass is
run separately by scripts/judge_hle.py and stored alongside). LiveBench graders follow the repo's process_results logic
in simplified form: MATH-style LaTeX string normalization for AMPS_Hard, list matching for web_of_lies / zebra."""

import re

ANSWER_RE = re.compile(r"^ANSWER:\s*(.+?)\s*$", re.MULTILINE)


def last_answer(output_text: str) -> str | None:
    m = ANSWER_RE.findall(output_text or "")
    return m[-1] if m else None


def _strip_latex(s: str) -> str:
    """Hendrycks MATH strip_string, abridged."""
    s = s.strip().replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    s = re.sub(r"\\boxed\{(.*)\}", r"\1", s)
    s = s.replace("tfrac", "frac").replace("dfrac", "frac").replace("\\left", "").replace("\\right", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "").replace("\\$", "").replace("$", "")
    s = re.sub(r"\\text\{\s*([^}]*)\}", r"\1", s)
    s = s.replace("\\%", "").replace("%", "")
    s = s.replace(" .", " 0.").replace("{.", "{0.")
    if s.startswith("."):
        s = "0" + s
    if "=" in s and len(s.split("=")[0]) <= 2:
        s = s.split("=")[1]
    s = re.sub(r"\\sqrt(\w)", r"\\sqrt{\1}", s)
    s = s.replace(" ", "")
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"^(\d+)/(\d+)$", r"\\frac{\1}{\2}", s)
    if s == "0.5":
        s = "\\frac{1}{2}"
    return s


def _num(s: str):
    t = s.replace(",", "").replace("$", "").strip().rstrip(".")
    t = re.sub(r"\s*(units?|dollars?)$", "", t, flags=re.IGNORECASE)
    try:
        return float(t)
    except ValueError:
        return None


def _norm_text(s: str) -> str:
    s = s.strip().strip("*`'\" ").rstrip(".").lower()
    s = re.sub(r"^\(?([a-z])\)?$", r"\1", s)  # (b) -> b
    return re.sub(r"\s+", " ", s)


def grade_hle(ans: str, truth: str, answer_type: str) -> bool:
    if ans is None:
        return False
    if answer_type == "multipleChoice":
        return _norm_text(ans)[:1] == _norm_text(truth)[:1] and len(_norm_text(ans).split()[0].rstrip(".):")) == 1
    a, t = _num(ans), _num(truth)
    if a is not None and t is not None:
        return abs(a - t) <= 1e-6 * max(1.0, abs(t))
    return _norm_text(ans) == _norm_text(truth) or _strip_latex(ans).lower() == _strip_latex(truth).lower()


def _latex_equiv(a: str, b: str) -> bool:
    """Symbolic equivalence via sympy's LaTeX parser (simplify(a-b)==0, then a numeric spot check)."""
    try:
        from sympy import N, simplify
        from sympy.parsing.latex import parse_latex

        ea, eb = parse_latex(_strip_latex(a).replace("\\!", "")), parse_latex(_strip_latex(b).replace("\\!", ""))
        d = simplify(ea - eb)
        if d == 0:
            return True
        vals = {s: 1.37 + 0.11 * i for i, s in enumerate(sorted(d.free_symbols, key=str))}
        return abs(complex(N(d.subs(vals)))) < 1e-8
    except Exception:  # noqa: BLE001  (parser raises many exception types on unparseable LaTeX)
        return False


def _items(s: str) -> list[str]:
    s = re.sub(r"</?solution>|\*\*", "", s)
    return [_norm_text(x) for x in s.split(",") if x.strip()]


def _latex_equiv_up_to(a: str, b: str, mode: str) -> bool:
    """mode 'constant': a-b is constant (indefinite integrals); 'sign': a == b or a == -b (characteristic polynomial convention)."""
    try:
        from sympy import diff, simplify
        from sympy.parsing.latex import parse_latex

        ea, eb = parse_latex(_strip_latex(a)), parse_latex(_strip_latex(b))
        if mode == "sign":
            return simplify(ea - eb) == 0 or simplify(ea + eb) == 0
        d = simplify(ea - eb)
        return all(simplify(diff(d, v)) == 0 for v in d.free_symbols) if d.free_symbols else True
    except Exception:  # noqa: BLE001
        return False


def grade_livebench(ans: str, truth: str, task: str, subtask: str | None = None) -> bool:
    if ans is None:
        return False
    if task == "AMPS_Hard":
        if _strip_latex(ans) == _strip_latex(truth) or _latex_equiv(ans, truth):
            return True
        if subtask == "amps_hard_integral":
            return _latex_equiv_up_to(re.sub(r"\s*\+\s*C$", "", ans.strip()), truth, "constant")
        if subtask == "amps_hard_characteristic_polynomial":
            return _latex_equiv_up_to(ans, truth, "sign")
        return False
    if task == "olympiad":  # LiveBench gives partial credit per position; here exact list match
        return [x.strip() for x in re.sub(r"[^\d,]", "", ans).split(",") if x.strip()] == [
            x.strip() for x in truth.split(",")
        ]
    if task == "math_comp":  # AMC prompts ask for the letter repeated five times ("DDDDD"); accept either form
        a, t = _norm_text(ans), _norm_text(truth)
        a = re.sub(r"^([a-e])\1{4}$", r"\1", a)
        t = re.sub(r"^([a-e])\1{4}$", r"\1", t)
        return (a.lstrip("0") or "0") == (t.lstrip("0") or "0")
    if task == "spatial":
        return _num(ans) is not None and _num(truth) is not None and _num(ans) == _num(truth)
    if task in ("web_of_lies_v2", "zebra_puzzle"):
        return _items(ans) == _items(truth)
    if task == "cta":
        return _norm_text(ans) == _norm_text(truth)
    raise ValueError(task)
