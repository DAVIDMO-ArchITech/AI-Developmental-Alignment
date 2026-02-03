# Repository Map

Quick reference for navigating the Developmental Alignment repository.

## Directory Structure

```
AI-Developmental-Alignment/
│
├── README.md                          # Project overview and quick start
├── LICENSE                            # Apache 2.0
├── NOTICE                             # Copyright notice
│
├── docs/                              # Human-readable documentation
│   ├── 00-executive-summary.md        # 1-page overview for stakeholders
│   ├── 01-theory-principles.md        # Full theoretical framework
│   ├── 02-assurance-metrics.md        # What we measure and why
│   ├── 03-governance-external-authority.md  # Policy and control model
│   └── 04-replication-guide.md        # How to reproduce results
│
├── spec/                              # Technical specifications
│   ├── holy-data-protocol.md          # Full HDP v0.1 spec
│   ├── data-bundle-schema.json        # JSON Schema for packets
│   ├── diff-format-spec.md            # How diffs are structured
│   ├── tag-taxonomy.md                # Reason code definitions
│   └── eval-spec.md                   # Evaluation methodology
│
├── experiments/                       # Simulations and results
│   ├── simulations/
│   │   └── developmental_alignment_sim.py  # Main simulation script
│   └── eval-results/                  # Output from runs
│
├── pipeline/                          # Data processing tools (planned)
│   ├── normalization/                 # Text normalization scripts
│   ├── diff-generator/                # Generate structured diffs
│   ├── bundle-validator/              # Validate packet integrity
│   └── exporters/                     # Export to training formats
│
├── data/                              # Training data (HDP bundles)
│   ├── raw/                           # Original source texts
│   ├── normalized/                    # After normalization pass
│   ├── bundles/                       # Closed segment packets
│   └── gold/                          # Validated, production-ready
│
├── src/                               # Core library (planned)
│   ├── core/                          # Base classes and utilities
│   ├── metrics/                       # Variance calculation, stats
│   └── gating/                        # Curriculum gate logic
│
├── claude_context/                    # AI assistant context package
│   ├── repo_map.md                    # This file
│   ├── canonical_intent.md            # Project ground truth
│   ├── data_bundle_example.jsonl      # Sample packets
│   ├── tag_taxonomy.md                # Symlink to spec/
│   └── eval_suite.md                  # Symlink to spec/
│
├── figures/                           # Visual assets
│   └── *.png, *.pdf                   # Diagrams, charts
│
├── archive/                           # Deprecated but preserved
│   └── old_code/                      # Previous implementations
│       └── README.md                  # Why it's archived
│
└── .github/                           # GitHub automation
    ├── CODEOWNERS                     # Who can merge where
    ├── workflows/
    │   └── validate.yml               # CI: schema validation
    └── ISSUE_TEMPLATE/
        ├── suggestion_a.yml           # Blind suggestion template
        └── suggestion_b.yml           # Blind suggestion template
```

## Key Files by Task

| Task | File(s) |
|------|---------|
| Understand the project | `README.md`, `claude_context/canonical_intent.md` |
| Run the simulation | `experiments/simulations/developmental_alignment_sim.py` |
| Read the HDP spec | `spec/holy-data-protocol.md` |
| Understand data format | `spec/data-bundle-schema.json`, `claude_context/data_bundle_example.jsonl` |
| Review evaluation method | `spec/eval-spec.md` |
| See reason code definitions | `spec/tag-taxonomy.md` |
| Check deprecated code | `archive/old_code/` |

## Entry Points

**For researchers:**
1. `docs/01-theory-principles.md` — Understand the framework
2. `spec/eval-spec.md` — Learn evaluation methodology
3. `experiments/simulations/` — Run and extend experiments

**For implementers:**
1. `spec/holy-data-protocol.md` — Full protocol spec
2. `spec/data-bundle-schema.json` — Data format
3. `pipeline/` — Processing tools (when available)

**For AI assistants:**
1. `claude_context/canonical_intent.md` — Ground truth
2. `claude_context/repo_map.md` — This file
3. `spec/` — Technical details
