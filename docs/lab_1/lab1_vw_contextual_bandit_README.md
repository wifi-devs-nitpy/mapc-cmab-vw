# Lab 1 — Minimal Online Contextual Bandit with Vowpal Wabbit

## Goal

Understand the complete online loop:

`context -> VW PMF -> sample action -> environment reward -> propensity feedback -> VW update`

## Environment

Two binary context features:

- `[0, 0]`
- `[0, 1]`
- `[1, 0]`
- `[1, 1]`

Three actions:

- A0
- A1
- A2

Reward table:

| Context | A0 | A1 | A2 |
|---|---:|---:|---:|
| [0,0] | 4 | 2 | 3 |
| [0,1] | 4 | 5 | 3 |
| [1,0] | 2 | 2 | 3 |
| [1,1] | 2 | 5 | 3 |

Known optimal policy:

- `[0,0] -> A0`
- `[0,1] -> A1`
- `[1,0] -> A2`
- `[1,1] -> A1`

## Install

```powershell
python -m pip install vowpalwabbit==9.11.2
```

Then:

```powershell
python lab1_vw_contextual_bandit.py
```

## Important

For this lab we use `--cb_explore 3 --epsilon 0.2`.

VW returns a probability distribution:

```text
[p(A0|x), p(A1|x), p(A2|x)]
```

We sample the action ourselves:

```text
action ~ PMF
```

If A1 was selected and the PMF was:

```text
[0.0667, 0.8667, 0.0667]
```

then the logged propensity is:

```text
p = 0.8667
```

That is the number returned to VW during learning.

The environment never supplies that number.

## Why reward becomes cost

VW's contextual-bandit feedback is expressed as cost, where lower is better.

So this experiment uses:

```text
cost = -reward
```

Thus reward 5 becomes cost -5.

## What to record

Study the first 15 interactions:

```text
context
PMF
chosen action
chosen probability
reward
cost
```

Then compare those against the known optimal policy.

---

## Official references

Vowpal Wabbit's current documentation describes `--cb_explore K` as online contextual-bandit exploration when the action set is fixed, and documents epsilon-greedy as taking the learned policy with probability `1-epsilon` and choosing uniformly at random with the remaining epsilon probability.

The current Python API exposes `PredictionType.ACTION_PROBS`, which returns a list of action probabilities.

The current PyPI release is 9.11.2 and provides CPython 3.14 wheels.
