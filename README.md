# Developmental Alignment

**Author:** David M. Orlo  
**Status:** Independent Research

---

## What This Is

AI alignment where behavioral invariants are embedded **during training**, not patched afterward.

**Core claim:** Training order matters. Phased training produces more stable alignment than mixed training + late correction.

**Proof:** ~4.5× variance reduction in refusal consistency across paraphrased unsafe prompts.

---

## Repository Structure

```
├── README.md                 ← You are here
├── THEORY.md                 ← Full theoretical framework
├── HOLY_DATA_PROTOCOL.md     ← Data curation spec (HDP v0.1)
├── simulation.py             ← Runnable proof-of-concept
├── INSTRUCTIONS.md           ← How to run the simulation
├── VERIFICATION.md           ← How to verify results
└── figures/                  ← Visual assets for papers
```

That's it. No nested folders. Everything important is at the top level.

---

## Quick Start

```bash
pip install torch transformers datasets peft scipy matplotlib
python simulation.py
```

Takes 30-60 min on CPU. Use `--device cuda` for GPU.

---

## The Experiment

**Variant A (Post-hoc):**
1. Train on mixed data (safe + refusal together)
2. Late refusal patch

**Variant B (Developmental):**
1. Safe-only training first
2. Refusal-only training second  
3. Mixed training last

**Result:** Variant B has ~4.5× lower variance on paraphrase tests.

| Metric | Post-Hoc | Developmental |
|--------|----------|---------------|
| Mean Variance | 0.045 | 0.010 |

Lower variance = more consistent = harder to jailbreak.

---

## Holy Data Protocol (HDP)

Training data curation with full audit trail:

1. **Original** → preserved exactly
2. **Normalized** → neutral rewrite by Team 1
3. **Blind suggestions** → A and B propose edits independently (can't see each other)
4. **Adjudication** → Team 2 votes, documents reasons
5. **Final** → closed packet with full lineage

See `HOLY_DATA_PROTOCOL.md` for the complete spec.

---

## Why This Matters

- **For labs:** Training schedules become an alignment lever
- **For researchers:** Curriculum order is experimentally tractable  
- **For policy:** Training methodology becomes auditable

---

## License

Apache 2.0
