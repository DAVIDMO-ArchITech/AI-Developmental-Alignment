# Tag Taxonomy — HDP Reason Codes

**Version:** HDP_RC_v1  
**Protocol:** Holy Data Protocol v0.1

## Purpose

Reason codes provide structured, auditable justification for every adjudication decision. They enable:

1. **Training signal extraction** — Weighted by disagreement patterns
2. **Audit trails** — Why each change was accepted or rejected
3. **Quality metrics** — Track distribution of change types over time
4. **Bias detection** — Flag over-concentration of certain codes

## Core Codes (Required Vocabulary)

These five codes cover 90%+ of legitimate edits. Use them as the primary vocabulary.

| Code | Definition | Example |
|------|------------|---------|
| `clarity` | Improves readability, reduces ambiguity, simplifies sentence structure | "The policy was implemented" → "We implemented the policy" |
| `factual_correction` | Fixes verifiable factual errors | "Founded in 1987" → "Founded in 1985" |
| `bias_reduction` | Removes loaded language, emotional framing, or hidden assumptions | "The radical proposal" → "The proposal" |
| `redundancy` | Eliminates unnecessary repetition or verbosity | "In order to" → "To" |
| `intent_preservation` | Maintains original meaning while improving form; used when other codes might suggest meaning changed | Structural rewrite that keeps semantics |

## Extension Codes (Use Sparingly)

Deploy only when core codes are insufficient. Overuse indicates scope creep.

| Code | Definition | When to Use |
|------|------------|-------------|
| `safety` | Content could cause harm if published as-is | Dangerous instructions, doxxing, etc. |
| `formatting` | Non-semantic formatting changes | Punctuation, capitalization, spacing |
| `style_consistency` | Aligns with established style guide | Terminology standardization |
| `scope_control` | Content exceeds or falls short of intended scope | Adding/removing topics |
| `ambiguity_resolution` | Clarifies genuinely ambiguous meaning | When multiple interpretations exist |

## Usage Rules

### 1. Minimum Necessary
Apply the fewest codes that accurately describe the change. One code is often sufficient.

### 2. No Free Text
Reason codes are the explanation. Do not add narrative justifications in the packet—codes must be self-documenting.

### 3. Multiple Codes = Complex Change
If a single change requires 3+ codes, consider splitting into smaller diffs.

### 4. Disagreement Weighting
When Suggester A and B disagree, the adjudication reason codes carry extra training weight. Document carefully.

## Anti-Patterns

| Bad Practice | Why It Fails | Fix |
|--------------|--------------|-----|
| Using `clarity` for everything | Loses signal value | Use precise codes |
| Adding `safety` without clear harm | Inflates false positives | Reserve for actual risk |
| Free-text explanations | Not machine-readable | Use codes only |
| Applying codes to unchanged text | Creates noise | Codes only on actual changes |

## Code Distribution Monitoring

Healthy distributions (rough targets):

| Code | Expected Frequency |
|------|-------------------|
| `clarity` | 30-50% |
| `factual_correction` | 5-15% |
| `bias_reduction` | 15-30% |
| `redundancy` | 10-20% |
| `intent_preservation` | 5-15% |
| Extension codes (all) | <10% |

Significant deviation warrants process review.

## Version History

- **HDP_RC_v1** — Initial taxonomy (2025)
