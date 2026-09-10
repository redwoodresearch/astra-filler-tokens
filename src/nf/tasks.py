"""Procedurally generated serial-depth tasks with a tunable depth d, fresh per (seed, depth, idx).

Every task yields a Problem whose `statement` is deterministic text and whose `answer` is a single short
token drawn from a small alphabet (`answer_spec` tells the model the alphabet, `chance` is 1/|alphabet|).
The *statement* is also what arms B/C repeat verbatim, so it must not contain the instructions."""

import random
from dataclasses import dataclass

N = 8  # permutation task alphabet {1..N}


@dataclass(frozen=True)
class Problem:
    problem_id: str
    task: str
    depth: int
    answer: str
    answer_spec: str
    chance: float
    statement: str


def _rng(task: str, depth: int, idx: int, seed: int) -> random.Random:
    # perm keeps its historical stream (f"{seed}-{depth}-{idx}") so main/fillerD problem ids stay valid.
    return random.Random(f"{seed}-{depth}-{idx}" if task == "perm" else f"{task}-{seed}-{depth}-{idx}")


def _pid(task: str, depth: int, idx: int, seed: int) -> str:
    return f"d{depth}_i{idx}_s{seed}" if task == "perm" else f"{task}_d{depth}_i{idx}_s{seed}"


# --- 1. composition of d random permutations of {1..8} --------------------------------------------
def perm(depth: int, idx: int, seed: int) -> Problem:
    rng = _rng("perm", depth, idx, seed)
    perms = [rng.sample(range(1, N + 1), N) for _ in range(depth)]
    start = rng.randint(1, N)
    x = start
    for p in perms:
        x = p[x - 1]
    lines = [f"Permutations of {{1,...,{N}}}:"]
    lines += [f"P{i + 1}: " + " ".join(f"{a}->{p[a - 1]}" for a in range(1, N + 1)) for i, p in enumerate(perms)]
    order = ", then ".join(f"P{i + 1}" for i in range(depth))
    lines.append(f"Start with the number {start}. Apply {order}. What number results?")
    return Problem(_pid("perm", depth, idx, seed), "perm", depth, str(x), "a single digit 1-8", 1 / N, "\n".join(lines))


# --- 2. iterated random lookup table f^d(x) on {0..9} ----------------------------------------------
def lookup(depth: int, idx: int, seed: int) -> Problem:
    rng = _rng("lookup", depth, idx, seed)
    while True:  # reject tables where the trajectory closes into a cycle too early (depth would be fake)
        f = [rng.randrange(10) for _ in range(10)]
        x0 = rng.randrange(10)
        traj, x = [x0], x0
        for _ in range(depth):
            x = f[x]
            traj.append(x)
        if len(set(traj)) >= min(depth + 1, 7):
            break
    table = ", ".join(f"f({a})={f[a]}" for a in range(10))
    st = f"A function f on {{0,...,9}} is defined by: {table}.\nStart with x = {x0}. Apply f {depth} time{'s' if depth > 1 else ''} (compute f(f(...f(x)...))). What is the result?"
    return Problem(_pid("lookup", depth, idx, seed), "lookup", depth, str(x), "a single digit 0-9", 0.1, st)


# --- 3. modular arithmetic chain mod 7 -------------------------------------------------------------
def modchain(depth: int, idx: int, seed: int) -> Problem:
    rng = _rng("modchain", depth, idx, seed)
    x = rng.randrange(7)
    steps, v = [], x
    for i in range(depth):
        op, a = rng.choice(["add", "subtract", "multiply by"]), rng.randint(2, 6)
        v = {"add": v + a, "subtract": v - a, "multiply by": v * a}[op] % 7
        steps.append(f"Step {i + 1}: {op} {a}.")
    st = (
        f"Start with the value {x}. Perform the following steps in order, reducing modulo 7 after every step "
        f"(so the value is always in 0-6).\n" + "\n".join(steps) + "\nWhat is the final value?"
    )
    return Problem(_pid("modchain", depth, idx, seed), "modchain", depth, str(v), "a single digit 0-6", 1 / 7, st)


# --- 4. d-hop reachability on a random digraph ----------------------------------------------------
def reach(depth: int, idx: int, seed: int) -> Problem:
    """Backbone path src=v0->v1->...->v_d guarantees a node at distance exactly d. Every node gets a level
    (backbone index, or parent level + 1 for side nodes) and extra edges only go to level <= own level + 1,
    so no edge can shorten a shortest path. 'yes' target: v_d (distance exactly d). 'no' target: a side node at
    level d+1 (reachable, but not within d hops). Node labels are shuffled letters."""
    rng = _rng("reach", depth, idx, seed)
    extras = 6
    n = depth + 1 + extras
    labels = rng.sample([chr(ord("A") + i) for i in range(26)], n)
    level = {v: v for v in range(depth + 1)}  # backbone nodes 0..d
    edges = {v: set() for v in range(n)}
    for v in range(depth):
        edges[v].add(v + 1)
    side = list(range(depth + 1, n))
    # one side node hangs off v_d (level d+1, the hard negative); the rest hang off random backbone nodes
    parents = [depth] + [rng.randrange(depth + 1) for _ in side[1:]]
    for w, u in zip(side, parents):
        edges[u].add(w)
        level[w] = level[u] + 1
    for _ in range(n // 2):  # extra non-shortening edges
        u = rng.randrange(n)
        cands = [v for v in range(n) if v != u and level[v] <= level[u] + 1 and v not in edges[u]]
        if cands:
            edges[u].add(rng.choice(cands))
    want_yes = idx % 2 == 0
    tgt = depth if want_yes else side[0]
    elist = [(labels[u], labels[v]) for u in edges for v in edges[u]]
    rng.shuffle(elist)
    st = (
        f"Directed graph on {n} nodes with edges (from->to):\n"
        + ", ".join(f"{a}->{b}" for a, b in elist)
        + f"\nCan node {labels[tgt]} be reached from node {labels[0]} by following at most {depth} "
        f"edge{'s' if depth > 1 else ''}? Answer yes or no."
    )
    return Problem(_pid("reach", depth, idx, seed), "reach", depth, "yes" if want_yes else "no", "yes or no", 0.5, st)


# --- 5. elementary cellular automaton, random rule, cyclic 12-cell tape, d steps -----------------
def ca(depth: int, idx: int, seed: int) -> Problem:
    rng = _rng("ca", depth, idx, seed)
    width, cell = 12, 5  # report cell 5 (1-indexed)
    rule = {f"{a}{b}{c}": rng.randrange(2) for a in (0, 1) for b in (0, 1) for c in (0, 1)}
    tape = [rng.randrange(2) for _ in range(width)]
    t = tape[:]
    for _ in range(depth):
        t = [rule[f"{t[(i - 1) % width]}{t[i]}{t[(i + 1) % width]}"] for i in range(width)]
    table = ", ".join(f"{k}->{v}" for k, v in rule.items())
    st = (
        f"A one-dimensional cellular automaton runs on a cyclic tape of {width} cells (cell {width} is adjacent to cell 1). "
        f"At each step every cell is updated simultaneously from the triple (left neighbour, itself, right neighbour) "
        f"using the rule: {table}.\nInitial tape (cells 1-{width}): {''.join(map(str, tape))}\n"
        f"After {depth} step{'s' if depth > 1 else ''}, what is the value of cell {cell}?"
    )
    return Problem(_pid("ca", depth, idx, seed), "ca", depth, str(t[cell - 1]), "a single digit 0 or 1", 0.5, st)


# --- 6. stack machine trace ----------------------------------------------------------------------
def stack(depth: int, idx: int, seed: int) -> Problem:
    rng = _rng("stack", depth, idx, seed)
    while True:
        st, ops, origin = [], [], []  # origin[i] = op index that pushed st[i]
        for i in range(depth):
            choices = ["push"] + (["pop"] if len(st) > 1 else []) + (["swap"] if len(st) >= 2 else [])
            if i == depth - 1:
                choices = [c for c in choices if c != "push"] or ["push"]
            op = rng.choice(choices)
            if op == "push":
                v = rng.randint(1, 9)
                st.append(v)
                origin.append(i)
                ops.append(f"push {v}")
            elif op == "pop":
                st.pop()
                origin.pop()
                ops.append("pop")
            else:
                st[-1], st[-2] = st[-2], st[-1]
                origin[-1], origin[-2] = origin[-2], origin[-1]
                ops.append("swap")
        if depth < 3 or depth - 1 - origin[-1] >= depth // 3:  # final top must have survived a good chunk of the trace
            break
    text = (
        f"A stack starts empty. Execute these {depth} operation{'s' if depth > 1 else ''} in order "
        f"(push v: put v on top; pop: remove the top element; swap: exchange the top two elements):\n"
        + "\n".join(f"{i + 1}. {o}" for i, o in enumerate(ops))
        + "\nWhat is the top element of the stack at the end?"
    )
    return Problem(_pid("stack", depth, idx, seed), "stack", depth, str(st[-1]), "a single digit 1-9", 1 / 9, text)


# --- 7. running XOR of d bits hidden among distractor fields ------------------------------------
def xor(depth: int, idx: int, seed: int) -> Problem:
    rng = _rng("xor", depth, idx, seed)
    recs, acc = [], 0
    for i in range(depth):
        b = rng.randrange(2)
        acc ^= b
        recs.append(
            f"record {i + 1}: flag={b} weight={rng.randint(10, 99)} code={rng.choice('ABCDEFGH')}{rng.randint(0, 9)}"
        )
    text = (
        f"Below are {depth} record{'s' if depth > 1 else ''}. Only the flag field matters.\n"
        + "\n".join(recs)
        + "\nWhat is the XOR (parity) of all the flag values, i.e. 1 if an odd number of flags are 1, else 0?"
    )
    return Problem(_pid("xor", depth, idx, seed), "xor", depth, str(acc), "a single digit 0 or 1", 0.5, text)


# --- Gen-Arithmetic, mirroring Greenblatt (2025) generate_arithmetic_problems.py: Python expression tree with
# `depth` operations, ops sampled from + - * // % with probs .3 .3 .15 .1 .05, leaves in [-99, 99], divisors in [-9, 9]\{0} ---
def arith(depth: int, idx: int, seed: int) -> Problem:
    rng = _rng("arith", depth, idx, seed)
    ops, probs = ["+", "-", "*", "//", "%"], [0.3, 0.3, 0.15, 0.1, 0.05]

    def sample(num_branches: int, top: bool = False) -> str:
        if num_branches == 0:
            return str(rng.randint(-99, 99))
        num_branches -= 1
        left = rng.randint(0, num_branches)
        left_e, right_e = sample(left), sample(num_branches - left)
        op = rng.choices(ops, probs)[0]
        if op in ("//", "%"):
            while eval(right_e) == 0:
                right_e = sample(num_branches - left)
        return f"{left_e} {op} {right_e}" if top else f"({left_e} {op} {right_e})"

    expr = sample(depth, top=True)
    val = eval(expr)
    st = f"Evaluate this Python expression. {expr}"
    return Problem(
        _pid("arith", depth, idx, seed),
        "arith",
        depth,
        str(val),
        "the final integer answer only (digits, no commas)",
        0.0,
        st,
    )


def _arith_shaped(shape: str):
    """Gen-Arithmetic with a controlled tree shape: 'chain' nests every op inside the next (height == ops, fully serial);
    'bal' splits ops evenly at every node (height ~ log2, maximally parallel). Same operator/operand distribution as arith."""

    def make(depth: int, idx: int, seed: int) -> Problem:
        rng = _rng(f"arith{shape}", depth, idx, seed)
        ops, probs = ["+", "-", "*", "//", "%"], [0.3, 0.3, 0.15, 0.1, 0.05]

        def sample(n: int, top: bool = False) -> str:
            if n == 0:
                return str(rng.randint(-99, 99))
            n -= 1
            left = (n if rng.random() < 0.5 else 0) if shape == "chain" else n // 2
            left_e, right_e = sample(left), sample(n - left)
            op = rng.choices(ops, probs)[0]
            if op in ("//", "%"):
                while eval(right_e) == 0:
                    right_e = sample(n - left)
            return f"{left_e} {op} {right_e}" if top else f"({left_e} {op} {right_e})"

        expr = sample(depth, top=True)
        return Problem(
            _pid(f"arith{shape}", depth, idx, seed),
            f"arith{shape}",
            depth,
            str(eval(expr)),
            "the final integer answer only (digits, no commas)",
            0.0,
            f"Evaluate this Python expression. {expr}",
        )

    return make


# --- non-toy benchmarks cached under results/data/<name>.jsonl (see README); depth is ignored, idx indexes the file ---
import functools
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_DATASETS: dict[str, list] = {}


@functools.cache
def _dataset(name: str):
    """Rows of data/<name>.jsonl (cached). HLE, AIME-Plus-Plus and LiveBench are not redistributed here: if the full file is
    absent, data/<name>_meta.jsonl (ids and metadata such as tier / task / category; no question text or answers) is
    used, which is enough for every analysis of the stored results. Rebuild the full files with
    scripts/build_hle_livebench.py and scripts/build_aimepp.py (needed only to run new evaluations)."""
    if name not in _DATASETS:
        path = DATA_DIR / f"{name}.jsonl"
        if not path.exists() and (DATA_DIR / f"{name}_meta.jsonl").exists():
            path = DATA_DIR / f"{name}_meta.jsonl"  # ids + metadata only; enough to analyse stored results
        _DATASETS[name] = [json.loads(line) for line in path.open()]
    return _DATASETS[name]


def _nhop(depth: int, idx: int, seed: int) -> Problem:
    r = _dataset(f"nhop_{depth}")[idx]
    return Problem(
        f"nhop_d{depth}_i{idx}_s{seed}", "nhop", depth, str(r["answer"]), r["answer_spec"], 0.0, r["statement"]
    )


def _nhopnl(depth: int, idx: int, seed: int) -> Problem:
    r = _dataset(f"nhopnl_{depth}")[idx]
    return Problem(
        f"nhopnl_d{depth}_i{idx}_s{seed}", "nhopnl", depth, str(r["answer"]), r["answer_spec"], 0.0, r["statement"]
    )


def _from_dataset(name: str):
    def make(depth: int, idx: int, seed: int) -> Problem:
        r = _dataset(name)[idx]
        return Problem(
            f"{name}_d{depth}_i{idx}_s{seed}",
            name,
            depth,
            str(r["answer"]),
            r["answer_spec"],
            float(r["chance"]),
            r["statement"],
        )

    return make


TASKS = {
    "perm": perm,
    "lookup": lookup,
    "modchain": modchain,
    "reach": reach,
    "ca": ca,
    "stack": stack,
    "xor": xor,
    "gsm8k": _from_dataset("gsm8k"),
    "gpqa": _from_dataset("gpqa"),
    "mmlupro": _from_dataset("mmlupro"),
    "arith": arith,
    "aime": _from_dataset("aime"),
    "aimepp": _from_dataset("aimepp"),
    "lehigh26": _from_dataset("lehigh26"),
    "arithchain": _arith_shaped("chain"),
    "arithbal": _arith_shaped("bal"),
    "hle": _from_dataset("hle"),
    "livebench": _from_dataset("livebench"),
    "nhop": _nhop,
    "nhopnl": _nhopnl,
}


def make_problem(depth: int, idx: int, seed: int = 0, task: str = "perm") -> Problem:
    return TASKS[task](depth, idx, seed)


def make_problems(depths: list[int], n: int, seed: int = 0, tasks: tuple[str, ...] = ("perm",)) -> list[Problem]:
    return [TASKS[t](d, i, seed) for t in tasks for d in depths for i in range(n)]


def dataset_row(name: str, idx: int) -> dict:
    """Raw dataset row for per-item metadata (e.g. HLE answer_type, LiveBench subtask)."""
    return _dataset(name)[idx]
