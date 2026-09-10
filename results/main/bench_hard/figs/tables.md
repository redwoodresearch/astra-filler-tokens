# Astra (prompted no-CoT) on HLE and LiveBench: full breakdown

Conditions: no filler; counting filler 1..300 (~600 tok) / 1..1000 (~2,000 tok); reasoning allowed at effort low. Gain and win/lose are paired per problem at counting 1000; p is an exact McNemar test.

## HLE (text-only, 2,158 questions)

Metric: LLM judge (official HLE grading). String-match numbers for the category level follow.

### Overall

| all | n | no filler | count 300 | count 1000 | reasoning low | gain @1000 | win/lose | p |
|---|---|---|---|---|---|---|---|---|
| HLE all | 2157 | 0.28 | 0.39 | 0.41 | 0.47 | +0.13 | 326/49 | 2.53e-51 |

### By category (judge)

| category | n | no filler | count 300 | count 1000 | reasoning low | gain @1000 | win/lose | p |
|---|---|---|---|---|---|---|---|---|
| Biology/Medicine | 222 | 0.34 | 0.37 | 0.40 | 0.42 | +0.06 | 20/6 | 0.00936 |
| Chemistry | 101 | 0.20 | 0.30 | 0.29 | 0.41 | +0.09 | 13/4 | 0.049 |
| Computer Science/AI | 224 | 0.27 | 0.34 | 0.37 | 0.42 | +0.09 | 27/6 | 0.000324 |
| Engineering | 64 | 0.14 | 0.17 | 0.20 | 0.22 | +0.06 | 5/1 | 0.219 |
| Humanities/Social Science | 193 | 0.36 | 0.45 | 0.47 | 0.42 | +0.10 | 26/6 | 0.000535 |
| Math | 975 | 0.29 | 0.44 | 0.46 | 0.56 | +0.17 | 182/18 | 2.57e-35 |
| Other | 176 | 0.18 | 0.31 | 0.34 | 0.43 | +0.16 | 31/3 | 7.66e-07 |
| Physics | 202 | 0.24 | 0.32 | 0.33 | 0.40 | +0.08 | 22/5 | 0.00151 |

### By category (string match)

| category | n | no filler | count 300 | count 1000 | reasoning low | gain @1000 | win/lose | p |
|---|---|---|---|---|---|---|---|---|
| Biology/Medicine | 222 | 0.32 | 0.34 | 0.37 | 0.38 | +0.05 | 14/4 | 0.0309 |
| Chemistry | 101 | 0.15 | 0.24 | 0.22 | 0.29 | +0.07 | 9/2 | 0.0654 |
| Computer Science/AI | 224 | 0.25 | 0.30 | 0.34 | 0.38 | +0.09 | 26/5 | 0.000192 |
| Engineering | 64 | 0.12 | 0.14 | 0.14 | 0.16 | +0.02 | 2/1 | 1 |
| Humanities/Social Science | 193 | 0.33 | 0.41 | 0.43 | 0.36 | +0.10 | 25/6 | 0.000878 |
| Math | 975 | 0.22 | 0.34 | 0.35 | 0.40 | +0.13 | 142/14 | 7.71e-28 |
| Other | 176 | 0.17 | 0.29 | 0.31 | 0.40 | +0.14 | 27/3 | 8.43e-06 |
| Physics | 202 | 0.14 | 0.17 | 0.17 | 0.21 | +0.02 | 9/4 | 0.267 |

### By category × answer type (judge)

| category | answer type | n | no filler | count 300 | count 1000 | reasoning low | gain @1000 | win/lose | p |
|---|---|---|---|---|---|---|---|---|---|
| Biology/Medicine | exactMatch | 75 | 0.15 | 0.19 | 0.20 | 0.27 | +0.05 | 7/3 | 0.344 |
| Biology/Medicine | multipleChoice | 147 | 0.44 | 0.46 | 0.50 | 0.50 | +0.07 | 13/3 | 0.0213 |
| Chemistry | exactMatch | 75 | 0.12 | 0.23 | 0.21 | 0.32 | +0.09 | 10/3 | 0.0923 |
| Chemistry | multipleChoice | 26 | 0.42 | 0.50 | 0.50 | 0.65 | +0.08 | 3/1 | 0.625 |
| Computer Science/AI | exactMatch | 158 | 0.15 | 0.22 | 0.25 | 0.34 | +0.10 | 20/4 | 0.00154 |
| Computer Science/AI | multipleChoice | 66 | 0.58 | 0.62 | 0.65 | 0.61 | +0.08 | 7/2 | 0.18 |
| Engineering | exactMatch | 39 | 0.05 | 0.10 | 0.15 | 0.18 | +0.10 | 4/0 | 0.125 |
| Engineering | multipleChoice | 25 | 0.28 | 0.28 | 0.28 | 0.28 | +0.00 | 1/1 | 1 |
| Humanities/Social Science | exactMatch | 114 | 0.26 | 0.37 | 0.37 | 0.35 | +0.11 | 17/5 | 0.0169 |
| Humanities/Social Science | multipleChoice | 79 | 0.51 | 0.57 | 0.61 | 0.52 | +0.10 | 9/1 | 0.0215 |
| Math | exactMatch | 886 | 0.27 | 0.42 | 0.44 | 0.55 | +0.17 | 168/15 | 6.56e-34 |
| Math | multipleChoice | 89 | 0.55 | 0.65 | 0.67 | 0.71 | +0.12 | 14/3 | 0.0127 |
| Other | exactMatch | 132 | 0.14 | 0.27 | 0.27 | 0.39 | +0.14 | 21/3 | 0.000277 |
| Other | multipleChoice | 44 | 0.30 | 0.45 | 0.52 | 0.55 | +0.23 | 10/0 | 0.00195 |
| Physics | exactMatch | 165 | 0.20 | 0.28 | 0.30 | 0.36 | +0.10 | 20/4 | 0.00154 |
| Physics | multipleChoice | 37 | 0.43 | 0.49 | 0.46 | 0.54 | +0.03 | 2/1 | 1 |

### By raw subject, n ≥ 15 (judge)

| subject | n | no filler | count 300 | count 1000 | reasoning low | gain @1000 | win/lose | p |
|---|---|---|---|---|---|---|---|---|
| Applied Mathematics | 98 | 0.21 | 0.35 | 0.35 | 0.46 | +0.13 | 13/0 | 0.000244 |
| Artificial Intelligence | 21 | 0.48 | 0.52 | 0.62 | 0.57 | +0.14 | 4/1 | 0.375 |
| Biochemistry | 16 | 0.44 | 0.44 | 0.50 | 0.56 | +0.06 | 1/0 | 1 |
| Biology | 31 | 0.32 | 0.29 | 0.29 | 0.39 | -0.03 | 0/1 | 1 |
| Chemistry | 92 | 0.21 | 0.30 | 0.32 | 0.42 | +0.11 | 13/3 | 0.0213 |
| Computer Science | 160 | 0.23 | 0.31 | 0.33 | 0.41 | +0.10 | 19/3 | 0.000855 |
| Ecology | 20 | 0.50 | 0.60 | 0.55 | 0.70 | +0.05 | 2/1 | 1 |
| Economics | 18 | 0.56 | 0.56 | 0.56 | 0.67 | +0.00 | 1/1 | 1 |
| Electrical Engineering | 26 | 0.08 | 0.08 | 0.12 | 0.12 | +0.04 | 1/0 | 1 |
| Genetics | 27 | 0.19 | 0.19 | 0.26 | 0.26 | +0.07 | 3/1 | 0.625 |
| History | 22 | 0.32 | 0.36 | 0.41 | 0.23 | +0.09 | 2/0 | 0.5 |
| Law | 20 | 0.55 | 0.55 | 0.60 | 0.55 | +0.05 | 1/0 | 1 |
| Linguistics | 37 | 0.32 | 0.41 | 0.49 | 0.46 | +0.16 | 6/0 | 0.0312 |
| Mathematics | 828 | 0.31 | 0.46 | 0.49 | 0.58 | +0.17 | 161/18 | 6.74e-30 |
| Medicine | 44 | 0.36 | 0.39 | 0.39 | 0.39 | +0.02 | 2/1 | 1 |
| Musicology | 17 | 0.29 | 0.35 | 0.41 | 0.47 | +0.12 | 2/0 | 0.5 |
| Neuroscience | 17 | 0.24 | 0.47 | 0.47 | 0.29 | +0.24 | 4/0 | 0.125 |
| Physics | 171 | 0.27 | 0.36 | 0.36 | 0.40 | +0.09 | 19/4 | 0.0026 |
| Trivia | 44 | 0.20 | 0.36 | 0.43 | 0.57 | +0.23 | 12/2 | 0.0129 |

## LiveBench (public 2024 releases, 618 answer-format questions)

Metric: programmatic grade (LiveBench convention; AMPS_Hard with symbolic equivalence).

### Overall

| all | n | no filler | count 300 | count 1000 | reasoning low | gain @1000 | win/lose | p |
|---|---|---|---|---|---|---|---|---|
| LiveBench all | 618 | 0.66 | 0.75 | 0.76 | 0.88 | +0.11 | 76/11 | 4.21e-13 |

### By category

| category | n | no filler | count 300 | count 1000 | reasoning low | gain @1000 | win/lose | p |
|---|---|---|---|---|---|---|---|---|
| data_analysis | 50 | 0.72 | 0.68 | 0.72 | 0.70 | +0.00 | 3/3 | 1 |
| math | 368 | 0.64 | 0.73 | 0.73 | 0.88 | +0.10 | 38/2 | 1.49e-09 |
| reasoning | 200 | 0.68 | 0.81 | 0.82 | 0.92 | +0.14 | 35/6 | 4.87e-06 |

### By task

| category | task | n | no filler | count 300 | count 1000 | reasoning low | gain @1000 | win/lose | p |
|---|---|---|---|---|---|---|---|---|---|
| data_analysis | cta | 50 | 0.72 | 0.68 | 0.72 | 0.70 | +0.00 | 3/3 | 1 |
| math | AMPS_Hard | 150 | 0.51 | 0.58 | 0.57 | 0.89 | +0.07 | 11/1 | 0.00635 |
| math | math_comp | 146 | 0.79 | 0.93 | 0.95 | 0.99 | +0.16 | 24/1 | 1.55e-06 |
| math | olympiad | 72 | 0.60 | 0.64 | 0.64 | 0.65 | +0.04 | 3/0 | 0.25 |
| reasoning | spatial | 50 | 0.68 | 0.82 | 0.82 | 0.86 | +0.14 | 8/1 | 0.0391 |
| reasoning | web_of_lies_v2 | 50 | 1.00 | 1.00 | 1.00 | 1.00 | +0.00 | 0/0 | 1 |
| reasoning | zebra_puzzle | 100 | 0.52 | 0.71 | 0.74 | 0.91 | +0.22 | 27/5 | 0.000113 |

### By subtask (where LiveBench defines one)

| category | task | subtask | n | no filler | count 300 | count 1000 | reasoning low | gain @1000 | win/lose | p |
|---|---|---|---|---|---|---|---|---|---|---|
| math | AMPS_Hard | amps_hard_characteristic_polynomial | 20 | 0.15 | 0.40 | 0.40 | 0.90 | +0.25 | 5/0 | 0.0625 |
| math | AMPS_Hard | amps_hard_complete_square | 10 | 1.00 | 1.00 | 1.00 | 1.00 | +0.00 | 0/0 | 1 |
| math | AMPS_Hard | amps_hard_derivatives | 10 | 0.80 | 0.80 | 0.70 | 0.90 | -0.10 | 0/1 | 1 |
| math | AMPS_Hard | amps_hard_determinant | 10 | 0.50 | 0.70 | 0.60 | 0.90 | +0.10 | 1/0 | 1 |
| math | AMPS_Hard | amps_hard_factor_polynomials | 10 | 0.80 | 0.90 | 0.90 | 0.90 | +0.10 | 1/0 | 1 |
| math | AMPS_Hard | amps_hard_gcd | 20 | 0.90 | 0.95 | 0.95 | 1.00 | +0.05 | 1/0 | 1 |
| math | AMPS_Hard | amps_hard_geometric_mean | 20 | 0.40 | 0.50 | 0.50 | 0.50 | +0.10 | 2/0 | 0.5 |
| math | AMPS_Hard | amps_hard_integral | 10 | 0.80 | 0.80 | 0.80 | 0.80 | +0.00 | 0/0 | 1 |
| math | AMPS_Hard | amps_hard_std | 20 | 0.20 | 0.20 | 0.20 | 1.00 | +0.00 | 0/0 | 1 |
| math | AMPS_Hard | amps_hard_variance | 20 | 0.20 | 0.20 | 0.25 | 1.00 | +0.05 | 1/0 | 1 |
| math | math_comp | aime_i_2024 | 14 | 0.57 | 0.71 | 0.79 | 1.00 | +0.21 | 3/0 | 0.25 |
| math | math_comp | aime_ii_2024 | 15 | 0.60 | 0.87 | 0.80 | 1.00 | +0.20 | 4/1 | 0.375 |
| math | math_comp | amc_12a_2023 | 25 | 0.84 | 0.96 | 1.00 | 1.00 | +0.16 | 4/0 | 0.125 |
| math | math_comp | amc_12b_2023 | 25 | 0.80 | 1.00 | 1.00 | 1.00 | +0.20 | 5/0 | 0.0625 |
| math | math_comp | smc | 17 | 0.94 | 0.94 | 0.94 | 1.00 | +0.00 | 0/0 | 1 |
| math | math_comp | updated_amc_12a_2023 | 25 | 0.84 | 0.96 | 1.00 | 1.00 | +0.16 | 4/0 | 0.125 |
| math | math_comp | updated_amc_12b_2023 | 25 | 0.80 | 0.96 | 0.96 | 0.96 | +0.16 | 4/0 | 0.125 |
| math | olympiad | imo | 29 | 0.59 | 0.66 | 0.66 | 0.69 | +0.07 | 2/0 | 0.5 |
| math | olympiad | usamo | 43 | 0.60 | 0.63 | 0.63 | 0.63 | +0.02 | 1/0 | 1 |
| reasoning | zebra_puzzle | level 08 | 3 | 1.00 | 1.00 | 1.00 | 1.00 | +0.00 | 0/0 | 1 |
| reasoning | zebra_puzzle | level 09 | 3 | 0.00 | 1.00 | 1.00 | 1.00 | +1.00 | 3/0 | 0.25 |
| reasoning | zebra_puzzle | level 12 | 3 | 0.67 | 0.67 | 0.67 | 1.00 | +0.00 | 0/0 | 1 |
| reasoning | zebra_puzzle | level 13 | 5 | 0.40 | 0.60 | 0.60 | 1.00 | +0.20 | 1/0 | 1 |
| reasoning | zebra_puzzle | level 14 | 4 | 0.50 | 0.75 | 0.75 | 0.75 | +0.25 | 1/0 | 1 |
| reasoning | zebra_puzzle | level 15 | 4 | 0.25 | 0.50 | 1.00 | 1.00 | +0.75 | 3/0 | 0.25 |
| reasoning | zebra_puzzle | level 16 | 10 | 0.50 | 0.60 | 0.70 | 0.80 | +0.20 | 2/0 | 0.5 |
| reasoning | zebra_puzzle | level 17 | 5 | 0.60 | 0.80 | 1.00 | 1.00 | +0.40 | 2/0 | 0.5 |
| reasoning | zebra_puzzle | level 18 | 3 | 0.33 | 0.67 | 0.67 | 1.00 | +0.33 | 2/1 | 1 |
| reasoning | zebra_puzzle | level 19 | 4 | 0.25 | 0.75 | 0.75 | 1.00 | +0.50 | 2/0 | 0.5 |
| reasoning | zebra_puzzle | level 20 | 6 | 0.00 | 0.50 | 0.17 | 0.83 | +0.17 | 1/0 | 1 |
