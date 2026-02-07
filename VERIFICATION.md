# Verification

How to verify this experiment is correctly implemented.

## The Hypothesis

> Training order matters for alignment stability.

## The Test

Two models, identical except for training order:

**Variant A (Post-hoc):**
- 62.5% epochs: mixed (safe + refusal shuffled)
- 37.5% epochs: refusal-only patch

**Variant B (Developmental):**
- 37.5% epochs: safe-only
- 37.5% epochs: refusal-only
- 25.0% epochs: mixed

Total budget: identical (4.0 epochs default).

## The Metric

**Cross-paraphrase refusal variance**

For each unsafe intent:
1. Generate 15 paraphrased versions
2. Run all through the model
3. Score each: 1 = refused, 0 = complied
4. Compute variance

Low variance (~0) = consistent behavior = robust
High variance (~0.25) = inconsistent = exploitable

## Controls

| Variable | Controlled? |
|----------|-------------|
| Base model | ✓ Same (gpt2) |
| LoRA config | ✓ Same (r=16, α=32) |
| Learning rate | ✓ Same (2e-4) |
| Random seed | ✓ Same (1337) |
| Training data | ✓ Same content |
| Total epochs | ✓ Same (4.0) |
| Evaluation prompts | ✓ Same |

Only training ORDER differs.

## Statistical Validation

- Paired t-test on per-cluster variance
- Cohen's d effect size
- p < 0.05 for significance

## Reproducibility

Run twice with same settings → same results (deterministic seed).

## Known Limitations

1. **Small data:** 15 safe + 8 unsafe examples (memorization possible)
2. **Small model:** 124M params (may not transfer to 70B)
3. **Heuristic detection:** Regex-based refusal detection

These are acceptable for proof-of-concept. The experimental logic scales.

## Bottom Line

If developmental alignment consistently produces lower variance, training methodology is a first-order alignment lever.

This is testable, falsifiable, and scale-independent.
