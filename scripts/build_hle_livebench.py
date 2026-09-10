"""Build data/hle.jsonl (text-only HLE) and data/livebench.jsonl (answer-format LiveBench tasks).

HLE: cais/hle test split, rows without an image, answer_type exactMatch or multipleChoice. LiveBench: public HF mirror
(only the retired 2024 releases are public; the current release is private), tasks whose answer is short:
math_comp, AMPS_Hard, spatial, web_of_lies_v2, zebra_puzzle, cta. Long-output tasks (tablejoin,
tablereformat) are excluded because the answer itself would be a paragraph."""

import json
import os
import re
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

OUT = Path(__file__).resolve().parents[1] / "data"


def hle():
    p = hf_hub_download(
        "cais/hle", "data/test-00000-of-00001.parquet", repo_type="dataset", token=os.environ.get("HF_TOKEN")
    )
    df = pd.read_parquet(p, columns=["id", "question", "image", "answer", "answer_type", "category", "raw_subject"])
    df = df[(df.image.fillna("") == "") & df.answer_type.isin(["exactMatch", "multipleChoice"])].reset_index(drop=True)
    rows = []
    for i, r in df.iterrows():
        mc = r.answer_type == "multipleChoice"
        spec = (
            "the letter of the correct answer choice"
            if mc
            else "the exact final answer only (a number, expression, or short phrase), with no units or explanation"
        )
        n_choices = len(re.findall(r"^[A-Z]\.\s", r.question, re.MULTILINE)) if mc else 0
        rows.append(
            {
                "id": r.id,
                "source": "cais/hle",
                "statement": r.question,
                "answer": r.answer,
                "answer_type": r.answer_type,
                "category": r.category,
                "subject": r.raw_subject,
                "answer_spec": spec,
                "chance": (1 / n_choices if n_choices else 0.0),
            }
        )
    (OUT / "hle.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    print("hle:", len(rows), df.answer_type.value_counts().to_dict())


SPEC = {
    "math_comp": "the final answer only (a three-digit integer for AIME-style problems, or the letter of the correct choice for AMC-style problems)",
    "AMPS_Hard": "the exact final answer in LaTeX (no \\boxed{}, no $)",
    "spatial": "a single integer",
    "web_of_lies_v2": "three words, yes or no, separated by commas (for example: yes, no, yes)",
    "zebra_puzzle": "the requested values, comma-separated, in the order asked (no <solution> tags)",
    "cta": "the single column class name",
    "olympiad": "the comma-separated list of expression identifiers, one per <missing X> tag in order (for example: 3,1,2)",
}


STRIP = {  # LiveBench's own CoT / output-format sentences, removed because they contradict the no-CoT `ANSWER:` protocol
    "AMPS_Hard": [
        (
            r"Please give an exact answer, and put your (final )?answer in latex( in terms of \$\\lambda\$)?( and)? in a \$\\boxed\{\}\$ \(for example, .*?\)\.",
            "Give an exact answer.",
        ),
        (r"Please put your final answer in a \$\\\\boxed\{\}\$\.", "Give an exact answer."),
        (r"Please put your final answer in a \$\\boxed\{\}\$\.", "Give an exact answer."),
    ],
    "math_comp": [
        (r"Please think step by step, and then display the answer at the very end of your response\. ", ""),
        (r" Remember to have the three digits as the last part of the response\.", ""),
        (
            r" Once you have your answer, please duplicate that letter five times in a single string\. For example, if the answer is F, then write FFFFF\.",
            "",
        ),
    ],
    "zebra_puzzle": [
        (r", in the following format: \*\*\*X\*\*\*, where X is the answer\.", "."),
        (r", in the following format: \*\*\*N\*\*\*, where N is the position\.", "."),
        (
            r"Think step by step and explain your reasoning, then output your answers in order in the format:\n<solution>.*?</solution>\nFor instance, .*?</solution>\n",
            "Answer the questions in order, comma-separated.\n",
        ),
    ],
    "spatial": [
        (
            r"Think step by step, and then put your answer in \*\*bold\*\* as a single (integer|phrase) \(for example, .*?\)\. ",
            "",
        )
    ],
    "web_of_lies_v2": [
        (
            r"Think step by step, and then put your answer in \*\*bold\*\* as a list of three words, yes or no \(for example, .*?\)\. ",
            "",
        )
    ],
    "cta": [],
    "olympiad": [
        (
            r"Your final answer should be STRICTLY in the format:\n\n<Detailed reasoning>\n\nAnswer: <comma separated list of numbers representing expression identifiers>",
            "",
        )
    ],
}


def strip_instructions(task: str, q: str) -> str:
    for pat, rep in STRIP[task]:
        q = re.sub(pat, rep, q, flags=re.DOTALL)
    return q


def livebench():
    rows = []
    for cat in ["math", "reasoning", "data_analysis"]:
        df = pd.read_parquet(
            hf_hub_download(f"livebench/{cat}", "data/test-00000-of-00001.parquet", repo_type="dataset")
        )
        df = df[df.task.isin(SPEC)]
        for _, r in df.iterrows():
            q = strip_instructions(r.task, r.turns[0])
            rows.append(
                {
                    "id": r.question_id,
                    "source": f"livebench/{cat}",
                    "task": r.task,
                    "subtask": r.get("subtask"),
                    "release": str(r.livebench_release_date)[:10],
                    "statement": q,
                    "answer": str(r.ground_truth),
                    "answer_spec": SPEC[r.task],
                    "chance": 0.0,
                }
            )
    rows.sort(
        key=lambda x: x["task"] == "olympiad"
    )  # olympiad was added later: keep it last so earlier indices are stable
    (OUT / "livebench.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    print("livebench:", len(rows), pd.Series([x["task"] for x in rows]).value_counts().to_dict())


if __name__ == "__main__":
    hle()
    livebench()
