# Teaching an LLM to Count Letters — GRPO + LoRA

This project teaches a language model to answer a question that sounds trivial but
that LLMs are famously bad at: **"How many times does the letter X appear in the
word Y?"**

## Why is that hard for an LLM?

A language model doesn't see words as letters — it sees them as *tokens* (chunks
of text). So when you ask it to count the g's in "engage", it can't actually look
at the letters; it just guesses from patterns it has seen. The base model in this
project confidently answered **"one g"** (the correct answer is 2).

## What was done

1. **Baseline with prompting only.** First, a Chain-of-Thought system prompt with
   one worked example was written, telling the model to spell the word letter by
   letter and keep a running count. This helps a lot — but it's not reliable.

2. **Reinforcement learning with GRPO.** To make the behavior reliable, the model
   was fine-tuned with GRPO (Group Relative Policy Optimization). The idea: for
   each question the model generates several candidate answers, each answer gets
   a score from hand-written **reward functions**, and the model is nudged toward
   whatever its better-scoring answers did. Five reward functions were written:

   | Reward function | What it checks |
   |---|---|
   | Numbering | The spelling steps are numbered 1, 2, 3, … in order |
   | Spelling  | The word is spelled with exactly the right letters |
   | Counting  | The running count is correct at every step |
   | Format    | The answer uses `<reasoning>…</reasoning><answer>…</answer>` tags |
   | Correctness | The final number is actually right |

3. **LoRA instead of full fine-tuning.** Training all 3 billion parameters of
   Qwen2.5-3B doesn't fit on a 16 GB GPU. Instead, small **LoRA adapters** were
   attached to the attention and MLP layers and only those were trained (the
   base model stays frozen). Training used Unsloth + vLLM on a single Tesla T4.

## Results

- Over a 200-step training run, the mean correctness reward climbed steadily
  (from ~1.69 to ~1.81 out of a max of 2.0), with ~94% of sampled answers
  correct late in training.
- **Before vs after** on the same question ("how many a's in *idea*"): the
  original model answered **3** (wrong); the fine-tuned model spells
  i-d-e-a with a correct running count and answers **1** (right).
- **No catastrophic forgetting:** asked "What is the capital of the
  Philippines?", both the original and the fine-tuned model still answer
  Manila — the LoRA adapter added a skill without erasing knowledge.

## Files

- `gen_ai_fundamentals_project_starter.ipynb` — the complete notebook with all
  code, training logs, reward plots, and before/after comparisons.

## Notes on running it

The notebook was run on a Tesla T4 (16 GB) in the course's Vocareum environment.
One environment fix was needed: the image shipped Triton 3.3.1, which crashes
while compiling vLLM's LoRA kernels on the T4 (an older GPU). The workaround —
documented in the notebook's second cell — pins Triton 3.2.0 via a locally
installed copy placed first on `sys.path`.

## Tech stack

Qwen2.5-3B-Instruct · GRPO (TRL) · LoRA (rank 64) · Unsloth · vLLM · Hugging Face Datasets
