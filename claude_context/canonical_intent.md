# Canonical Intent — Developmental Alignment Project

**Purpose:** Single-source-of-truth for AI assistants reviewing this repository.

## What This Project Is

A framework for training AI systems where **alignment is formative, not corrective**. The thesis: training order materially affects behavioral stability, and embedding invariants early reduces downstream assurance cost.

### Two Interconnected Components

1. **Developmental Alignment Theory** — The training paradigm
2. **Holy Data Protocol (HDP)** — The data curation system that feeds it

### The Core Claim

> Developmental alignment reduces cross-paraphrase refusal variance versus post-hoc alignment. This is an **assurance claim**, not a performance claim.

## What Success Looks Like

1. **Simulation** — Reproducible proof-of-concept showing variance reduction
2. **Protocol** — Complete HDP spec that can be implemented on GitHub
3. **Documentation** — Clear enough for replication and extension
4. **Adoption path** — Structure that supports unfunded → funded transition

## Key Principles (Non-Negotiable)

| Principle | Meaning |
|-----------|---------|
| **Formative, not corrective** | Embed invariants during early learning, not after capability emergence |
| **Training order matters** | Curriculum structure is a first-order alignment variable |
| **Provenance control** | Training data has full audit trail (HDP) |
| **External authority** | Governance stays external; no self-authorization |
| **Assurance over capability** | Variance reduction > raw performance |

## Data Bundle Format (HDP)

Every training example should eventually be traceable through:

```
original_text → normalized_text → diffs (A + B) → adjudication → final_text
```

With:
- Reason codes per decision
- Hashes for integrity
- Pseudonymous provenance
- Append-only corrections

See `spec/holy-data-protocol.md` for the full specification.

## Simulation Logic

**Variant A (Post-hoc):**
1. Mixed training (safe + refusal together)
2. Late refusal patch

**Variant B (Developmental):**
1. Safe-only training (establish helpful behavior)
2. Refusal-only training (establish safety boundary)
3. Mixed training (integration after stability)

**Evaluation:** Cross-paraphrase refusal variance. Lower = more stable = better.

## What NOT to Do

- Do not conflate this with Constitutional AI, RLHF, or DPO — different mechanism
- Do not claim this "solves" alignment — it's a structural risk reduction
- Do not remove old_code/ — preserve history, even if deprecated
- Do not publish assurance claims without statistical backing

## Directory Structure Rationale

```
/docs           — Human-readable theory and guides
/spec           — Machine-readable specs (HDP, schemas, eval)
/experiments    — Simulations and results
/pipeline       — Data processing tools (planned)
/data           — Actual training bundles (future)
/src            — Core library code (planned)
/claude_context — This folder; AI-assistant context package
/archive        — Deprecated code, preserved
/.github        — Automation and governance
```

## When Restructuring

If asked to restructure this repo:

1. Preserve all existing content (move, don't delete)
2. Follow the directory structure above
3. Keep `old_code/` in `archive/` with README
4. Ensure simulation can still run from new location
5. Update all relative paths in documentation

## For AI Assistants

When working on this project:

1. **Read this file first** — It's the ground truth
2. **Read relevant specs** — `spec/holy-data-protocol.md`, `spec/eval-spec.md`
3. **Preserve lineage** — Don't delete history; corrections create new artifacts
4. **Maintain role separation** — Content creation and commit authority are separate

## Contact

**Author:** David M. Orlo  
**Status:** Independent Researcher, Unfunded

This project is prepared for evaluation, replication, and discussion.
