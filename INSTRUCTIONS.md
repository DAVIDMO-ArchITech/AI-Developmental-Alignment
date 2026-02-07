# Running the Simulation

## Requirements

Python 3.9+ with:
```bash
pip install torch transformers datasets peft scipy matplotlib
```

## Quick Run

```bash
python simulation.py
```

Takes 30-60 minutes on CPU.

## GPU Run

```bash
python simulation.py --device cuda --fp16
```

Takes ~10 minutes.

## Options

```
--model         Base model (default: gpt2)
--device        cuda or cpu
--total_epochs  Training budget (default: 4.0)
--paraphrases   Paraphrases per unsafe intent (default: 15)
--fp16          Use mixed precision (GPU only)
--outdir        Output directory (default: results/)
```

## What It Does

1. **Trains Variant A (Post-hoc):**
   - Mixed training (safe + refusal together)
   - Late refusal patch

2. **Trains Variant B (Developmental):**
   - Safe-only training
   - Refusal-only training
   - Mixed integration

3. **Evaluates both:**
   - Generates paraphrases of unsafe prompts
   - Measures refusal consistency (variance)
   - Compares statistically

## Outputs

```
results/
├── results.json      # Numerical results
└── variance_plot.png # Visual comparison
```

## Expected Results

| Variant | Mean Variance |
|---------|---------------|
| Post-hoc | ~0.04-0.05 |
| Developmental | ~0.01 |

Variance ratio: ~4-5x improvement.

Lower variance = more consistent = more robust alignment.
