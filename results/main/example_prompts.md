# Example filler prompts (Astra, prompted no-CoT)

Every call sends two messages to the Responses API with `reasoning.effort=low` and a tight `max_output_tokens` cap. The developer message is the same for all arms:

```
You must not think, reason, plan, or use any hidden chain of thought before or while answering. Your reasoning must be completely empty. Produce your final output immediately and directly. Never write any working, explanation, or commentary anywhere in your output.
```

The user message carries the format instruction, the problem, and the filler. Three examples, verbatim from `nf.prompts.build`; only the counting run in the third example is abridged, and the omitted span is noted.

## Gen-Arithmetic (15 ops), 300 dots after the statement (arm XB, k=300)

Answer: `37945`. Filler tokens (o200k): 300. Expected output: `ANSWER: <n>` only.

```
The problem statement below is followed by exactly 300 periods separated by single spaces, written as 30 lines of ten periods each. These filler periods carry no information; ignore them. Respond with exactly one line of the form `ANSWER: <n>` where <n> is the final integer answer only (digits, no commas). Output nothing else: no working, no explanation, no repetition of the problem.

Evaluate this Python expression. ((63 + -61) + 96) - (((-66 + 97) - ((98 - (-63 + -60)) * (80 + 91))) + (((26 // -4) - 79) + (8 // (-65 - -37))))

. . . . . . . . . .
. . . . . . . . . .
. . . . . . . . . .
. . . . . . . . . .
. . . . . . . . . .
. . . . . . . . . .
. . . . . . . . . .
. . . . . . . . . .
. . . . . . . . . .
. . . . . . . . . .
. . . . . . . . . .
. . . . . . . . . .
. . . . . . . . . .
. . . . . . . . . .
. . . . . . . . . .
. . . . . . . . . .
. . . . . . . . . .
. . . . . . . . . .
. . . . . . . . . .
. . . . . . . . . .
. . . . . . . . . .
. . . . . . . . . .
. . . . . . . . . .
. . . . . . . . . .
. . . . . . . . . .
. . . . . . . . . .
. . . . . . . . . .
. . . . . . . . . .
. . . . . . . . . .
. . . . . . . . . .
```

## AIME-Plus-Plus (AIME tier), question repeated 5 times (arm RB, k=5)

Answer: `72`. Filler tokens (o200k): 279. Expected output: `ANSWER: <n>` only.

```
The problem statement is written out 6 times below; every copy is identical. Respond with exactly one line of the form `ANSWER: <n>` where <n> is the final integer answer only (digits, no units, no commas). Output nothing else: no working, no explanation, no repetition of the problem.

Let $X,Y,Z>1$ and $W>1$ satisfy $\log_X W=54$, $\log_Y W=24$, and $\log_{X^{3}Y^{2}Z^{2}}W=6$. Determine $\log_Z W$.

Let $X,Y,Z>1$ and $W>1$ satisfy $\log_X W=54$, $\log_Y W=24$, and $\log_{X^{3}Y^{2}Z^{2}}W=6$. Determine $\log_Z W$.

Let $X,Y,Z>1$ and $W>1$ satisfy $\log_X W=54$, $\log_Y W=24$, and $\log_{X^{3}Y^{2}Z^{2}}W=6$. Determine $\log_Z W$.

Let $X,Y,Z>1$ and $W>1$ satisfy $\log_X W=54$, $\log_Y W=24$, and $\log_{X^{3}Y^{2}Z^{2}}W=6$. Determine $\log_Z W$.

Let $X,Y,Z>1$ and $W>1$ satisfy $\log_X W=54$, $\log_Y W=24$, and $\log_{X^{3}Y^{2}Z^{2}}W=6$. Determine $\log_Z W$.

Let $X,Y,Z>1$ and $W>1$ satisfy $\log_X W=54$, $\log_Y W=24$, and $\log_{X^{3}Y^{2}Z^{2}}W=6$. Determine $\log_Z W$.
```

## n-hop natural facts (5 hops, rewritten phrasing), counting filler 1..300 (arm CB, k=300)

Answer: `Marilyn Buferd`. Filler tokens (o200k): 603. Expected output: `ANSWER: <n>` only.

```
After the problem, there will be filler tokens (counting from 1 to 300) to give you extra space to process the problem before answering. Respond with exactly one line of the form `ANSWER: <n>` where <n> is the answer only (a name, a US state, an element, a motto/flower, or a number). Output nothing else: no working, no explanation, no repetition of the problem.

Who is the Miss America winner for the 20th-century year whose last two digits equal the number of county-equivalents in the US state whose position in the order of joining the union equals the day-of-month of the birth of the Best Supporting Actress winner at the 30th Academy Awards?

Filler: 1 2 3 4 5 6 7 8 9 10 [... 11 through 296 omitted ...] 297 298 299 300
```
