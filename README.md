# Developmental Alignment

**Author:** David M. Orlo  
**Status:** Independent Research / Unfunded

## What This Is

A framework for training AI systems where alignment is **formative, not corrective**. Behavioral invariants are embedded during early representation learning—not patched after capability emergence.

This repo contains:
- **Theory & principles** for developmental alignment
- **Proof-of-concept simulation** comparing post-hoc vs developmental training
- **Holy Data Protocol** specification for provenance-controlled training data curation
- **Evaluation scaffolding** for measuring alignment robustness

## Core Hypothesis

> **Training order matters for alignment stability.**

Two models with identical capacity and compute, trained on the same data, exhibit different behavioral stability depending on **when** they encounter different data types.

| Metric | Post-Hoc Alignment | Developmental Alignment |
|--------|-------------------|------------------------|
| Mean Refusal Variance | 0.045 | 0.010 |
| Stability Improvement | baseline | ~4.5× more stable |

This is an **assurance claim**, not a performance claim.

## Core Principles

1. **Formative Alignment** — Behavioral invariants embedded during representation formation
2. **Curriculum Gating** — Exposure to complex/unsafe domains gated by demonstrated stability
3. **Provenance Control** — Training data curated with full audit trail (see Holy Data Protocol)
4. **External Authority** — Governance and policy enforcement remain external, never self-authorized
5. **Assurance-First Metrics** — Variance reduction and stability prioritized over raw capability

## Quick Start

```bash
# Install dependencies
pip install torch transformers datasets peft scipy matplotlib

# Run the simulation (CPU, ~30-60 min)
python experiments/simulations/developmental_alignment_sim.py

# Or with GPU
python experiments/simulations/developmental_alignment_sim.py --device cuda --fp16
```

## Repository Structure

```
├── docs/                           # Theory and documentation
│   ├── 00-executive-summary.md
│   ├── 01-theory-principles.md
│   ├── 02-assurance-metrics.md
│   ├── 03-governance-external-authority.md
│   └── 04-replication-guide.md
├── spec/                           # Technical specifications
│   ├── holy-data-protocol.md       # Full HDP spec
│   ├── data-bundle-schema.json
│   ├── diff-format-spec.md
│   ├── tag-taxonomy.md
│   └── eval-spec.md
├── experiments/                    # Simulations and eval results
│   ├── simulations/
│   └── eval-results/
├── pipeline/                       # Data processing tools (planned)
│   ├── normalization/
│   ├── diff-generator/
│   ├── bundle-validator/
│   └── exporters/
├── data/                           # Training data (bundles)
│   ├── raw/
│   ├── normalized/
│   ├── bundles/
│   └── gold/
├── src/                            # Core library
│   ├── core/
│   ├── metrics/
│   └── gating/
├── claude_context/                 # Context package for AI assistants
│   ├── repo_map.md
│   ├── canonical_intent.md
│   ├── data_bundle_example.jsonl
│   ├── tag_taxonomy.md
│   └── eval_suite.md
├── figures/                        # Visual assets
├── archive/                        # Deprecated code (preserved for history)
└── .github/                        # GitHub automation
    ├── CODEOWNERS
    ├── workflows/
    └── ISSUE_TEMPLATE/
```

## Holy Data Protocol (HDP)

Training data is curated through a multi-party, blind-review process:

1. **Team 1** — Normalizes text using dictionary/thesaurus constraints (no opinion, just semantic clarity)
2. **Users A & B** — Independently suggest edits (cannot communicate or see each other's work)
3. **Team 2** — Reviews original + normalized + both suggestion sets; majority vote decides

Output: **Closed Segment Packets** with full lineage (original → normalized → diffs → final → reason codes)

See `spec/holy-data-protocol.md` for the complete specification.

## Why This Matters

Current alignment approaches:
- Train on everything, patch later
- High variance under semantic perturbation
- Expensive red-teaming and audit cycles
- Brittle to adversarial prompts

Developmental alignment:
- Staged exposure, gated by stability
- Lower variance = more predictable behavior
- Audit trail built into training data
- Structural robustness, not reactive patches

## Status

This is an unfunded research project prepared for evaluation, replication, and discussion. The simulation is scale-independent—the same logic applies to larger models and datasets.

## License

Apache 2.0 — See LICENSE file.

## Citation

```bibtex
@misc{orlo2025developmental,
  author = {Orlo, David M.},
  title = {Developmental Alignment: Formative Training for High-Assurance AI Systems},
  year = {2025},
  url = {https://github.com/DAVIDMO-ArchITech/AI-Developmental-Alignment}
}
```
