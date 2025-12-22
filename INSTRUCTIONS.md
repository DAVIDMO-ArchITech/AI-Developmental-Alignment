\# Instructions: Running the Developmental Alignment Experiment



\*\*Author:\*\* David M. Orlo  

\*\*Role:\*\* Independent Researcher  



This document explains how to run the Developmental Alignment simulation included in this repository.  

The experiment compares \*\*post-hoc alignment\*\* against \*\*developmental (curriculum-gated) alignment\*\* using matched training budgets and variance-based evaluation.



The goal is to measure \*\*alignment robustness\*\*, not benchmark performance.



---



\## Overview



The experiment tests a single hypothesis:



> \*\*Training order matters for alignment stability.\*\*



Two models with identical capacity and compute are trained:

\- Variant A: Standard post-hoc alignment

\- Variant B: Developmental (phased) alignment



Evaluation measures \*\*cross-paraphrase refusal variance\*\* under unsafe prompts.



---



\## Prerequisites



\- Python 3.9 or newer

\- CPU or GPU (GPU recommended for faster runs)



Install required dependencies:



```bash

pip install torch transformers datasets peft matplotlib







```



Basic Run (CPU, Default Settings)



Runs the full experiment on CPU using default parameters.



python fixed\_alignment.py





Recommended for:



Verification



Reproducibility checks



Low-resource environments



Full Run (GPU, Higher Fidelity)



Runs the experiment on GPU with increased paraphrase coverage and mixed-precision training.



python fixed\_alignment.py --device cuda --paraphrases 20 --total\_epochs 6 --fp16





Recommended for:



Stronger statistical signal



Clearer variance separation



Review or presentation artifacts



Quick Test (Smaller Model)



Uses a smaller base model for faster iteration.



python fixed\_alignment.py --model distilgpt2 --total\_epochs 2





Recommended for:



Sanity checks



Debugging



CI-style validation



What the Code Does

Setup Phase



Loads a pre-trained language model (default: GPT-2, 124M parameters)



Applies LoRA adapters for efficient fine-tuning



Creates three datasets:



Safe: helpful, harmless queries



Refusal: unsafe queries with refusal responses



Mixed: safe + refusal combined and shuffled



Training Phase



Variant A — Post-Hoc Alignment



Mixed training (safe + refusal together)



Late-stage refusal-only patch



Variant B — Developmental Alignment



Safe-only training (establish helpful behavior)



Refusal-only training (establish safety boundary)



Mixed training (integration after stability)



Both variants use equal total training budgets.



Evaluation



Generates multiple paraphrases per unsafe intent



Runs all paraphrases through both models



Uses regex-based detection to classify responses:



1 = refused



0 = complied



Computes variance across paraphrases for each intent



Lower variance indicates more stable, robust alignment.



Outputs



Each run produces:



results.json



Per-intent refusal variance



Mean variance per variant



Summary statistics



variance\_plot.png



Visual comparison of alignment stability



Console Output



Mean variance comparison



Clear indication of which variant is more stable



Reproducibility Notes



Deterministic random seeds are used by default



Evaluation is deterministic (do\_sample = False)



Only the training order differs between variants



Notes on Scope



This experiment is intentionally scale-independent.



It does not claim to solve alignment.

It demonstrates a structural effect: formative training reduces behavioral instability earlier than post-hoc correction.



The same logic applies to larger models and datasets.



Status



This repository supports an unfunded research project prepared for evaluation, replication, and discussion.



