# Archive — Deprecated Code

This directory contains code that has been superseded but is preserved for historical reference.

## Why Keep This?

The Holy Data Protocol principle: **history never disappears, only accumulates**.

Even deprecated code may contain:
- Design decisions worth understanding
- Edge cases that informed current implementations
- Reproducibility requirements for prior results

## Contents

### `old_code/`

Previous implementations of the developmental alignment simulation:

- `fixed_alignment.py` — Earlier version with different random seed
- `fixed_alignment_sim.py` — Intermediate version
- `old_fixed_alignment_sim.py` — Original implementation

**Key differences from current version:**
- Different random seeds
- Slightly different training schedules
- Earlier refusal detection patterns

## Do Not Delete

These files should remain in the repository. If you need to remove them:

1. Create a git tag at the current commit
2. Document why removal is necessary
3. Ensure the files remain accessible via git history

## Current Implementation

The active simulation code is located at:

```
experiments/simulations/developmental_alignment_sim.py
```

Use that for all new experiments.
