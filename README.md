# Developmental Alignment

**Author:** David M. Orlo  
**Role:** Independent Researcher  

## Overview

Developmental Alignment is a theory-driven framework for training large language models (LLMs) as **governed, high-assurance systems** by embedding behavioral invariants during early representation learning rather than applying post-hoc constraints after capability emergence.

The core hypothesis is that **training order matters**: formative, curriculum-gated exposure collapses behavioral instability earlier, reducing downstream assurance cost, adversarial brittleness, and audit complexity.

This repository contains conceptual materials, simulations, and evaluation scaffolding supporting that claim.

---

## Motivation

Current LLM alignment approaches typically follow a two-stage pattern:

1. Broad, unconstrained pretraining  
2. Late-stage alignment via RLHF, filters, or policy overlays  

This introduces internal–external dissonance, manifesting as:
- Prompt-sensitive refusal instability
- Jailbreakability
- Latent capability leakage
- Increased red-teaming and patching costs

Developmental Alignment reframes alignment as a **formative property**, not a corrective one.

---

## Core Principles

- **Formative Alignment:** Behavioral invariants are embedded during representation formation.
- **Curriculum Gating:** Exposure to complex or unsafe domains is gated by demonstrated stability.
- **Provenance Control:** Early training data is curated and iteratively refined.
- **External Authority:** Governance and policy enforcement remain external and non-self-authorized.
- **Assurance-First Metrics:** Stability and variance reduction are prioritized over raw capability gains.

---

## Proof-of-Concept Simulation

The repository includes a small-scale, scale-independent simulation demonstrating the core effect:

- Two matched models are trained with identical capacity and data.
- Variant A applies post-hoc alignment.
- Variant B applies developmental (gated) alignment.
- Evaluation measures **cross-paraphrase refusal variance**.

### Representative Result

| Variant | Mean Refusal Variance |
|------|------------------------|
| Post-Hoc Alignment | 0.045231 |
| Developmental Alignment | 0.009876 |

**Result:** Developmental alignment reduces instability by ~4.5× under semantic perturbation.

This is not a performance claim.  
It is an **assurance claim**.

---

## Quick Start 

# Install dependencies
pip install torch transformers datasets peft scipy matplotlib

# Run the experiment (CPU mode, will take ~30-60 min)
python developmental_alignment.py

# Or GPU mode if you have CUDA
python developmental_alignment.py --device cuda --fp16

