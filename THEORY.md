# Developmental Alignment — Theory

## The Problem

Current alignment: train on everything → patch later (RLHF, filters, guardrails).

This is structurally brittle:
- Models pass safety tests, fail on paraphrased versions
- Jailbreaks exploit the gap between learned representations and overlaid constraints
- Red-teaming costs scale exponentially with capability

**Root cause:** Alignment signals arrive after representations consolidate. Late corrections fight established structure.

---

## The Insight

**Training order matters.**

Learning dynamics (biological and artificial):
1. Early representations are sticky
2. Curriculum effects are real
3. Stability precedes capability

If harmful content is present during early training, representations of it become foundational. Post-hoc alignment then fights against foundations rather than building on clean ones.

---

## The Framework

### Definition

**Developmental Alignment:** Behavioral invariants embedded during early representation formation, with exposure to risky domains gated by demonstrated stability.

### Five Principles

| Principle | Meaning |
|-----------|---------|
| **Formative Alignment** | Teach "what to do" before encountering content that tests boundaries |
| **Curriculum Gating** | Advance only when stability thresholds are met |
| **Provenance Control** | Training data has full audit trail (HDP) |
| **External Authority** | Safety rules are external specs, not learned heuristics |
| **Assurance-First** | Variance reduction > raw performance |

---

## Comparison

| Approach | When Alignment Happens | Gating |
|----------|----------------------|--------|
| RLHF | Post-training | None |
| Constitutional AI | Post-training | None |
| DPO | Post-training | None |
| **Developmental** | During training | Stability-gated |

Not mutually exclusive—developmental can be followed by RLHF refinement.

---

## Falsifiable Predictions

1. **Lower variance on paraphrase tests** ✓ (tested in simulation)
2. Reduced jailbreak susceptibility
3. Faster post-hoc alignment convergence
4. Stability transfers across domains

---

## Limitations

- **Scale:** Tested at 124M params. Unknown if effects hold at 70B+.
- **Curriculum design:** Optimal progression is an open question.
- **Metrics:** Cross-paraphrase variance is one measure; others may be better.

---

## The Claim

This doesn't "solve" alignment. It reduces a specific risk class—behavioral instability under semantic perturbation—in a measurable, reproducible way.
