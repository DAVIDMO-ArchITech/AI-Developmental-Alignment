# Holy Data Protocol (HDP) v0.1

Bias-resistant training data curation with immutable audit trails.

---

## Purpose

Create training data with full traceability: original → normalized → blind suggestions → adjudication → final. Every decision documented. History never deleted.

---

## Roles

| Role | Does | Cannot |
|------|------|--------|
| **Author** | Creates original material | Touch database |
| **Normalizer (Team 1)** | Rewrites for neutrality | See suggestions or final decisions |
| **Suggester A** | Proposes edits | See B's work or communicate with B |
| **Suggester B** | Proposes edits | See A's work or communicate with A |
| **Adjudicator (Team 2)** | Reviews all, votes on changes | Edit database directly |
| **Transcriber** | Enters closed packets | Modify content |

**Hard rule:** No one can both influence content AND commit records.

---

## Workflow

```
Original Text
     ↓
[Team 1: Normalize]
     ↓
Normalized Text
     ↓
┌────┴────┐
↓         ↓
[A edits] [B edits]  ← blind, independent
↓         ↓
└────┬────┘
     ↓
[Team 2: Adjudicate]
     ↓
Final Text + Decision Log
     ↓
[Transcriber: Commit]
     ↓
Closed Packet (immutable)
```

---

## Reason Codes

Tags for every adjudication decision:

| Code | Use When |
|------|----------|
| `clarity` | Improves readability |
| `factual_correction` | Fixes factual error |
| `bias_reduction` | Removes loaded language |
| `redundancy` | Cuts unnecessary repetition |
| `intent_preservation` | Keeps meaning while improving form |

No free-text explanations. Codes must be self-documenting.

---

## Closed Packet Format

Every committed packet contains:

```json
{
  "identity": {
    "segment_id": "seg_0001",
    "created_utc": "2025-01-15T10:00:00Z",
    "closed_utc": "2025-01-15T14:30:00Z"
  },
  "text_lineage": {
    "original_text": "...",
    "normalized_text": "...",
    "final_text": "..."
  },
  "suggestions": {
    "A_diffs": [...],
    "B_diffs": [...]
  },
  "adjudication": {
    "decisions": [...],
    "reason_codes": [...]
  },
  "integrity": {
    "hash_original": "sha256...",
    "hash_final": "sha256...",
    "hash_packet": "sha256..."
  },
  "provenance": {
    "normalizer_ref": "pseudonymous_id",
    "suggester_A_ref": "pseudonymous_id",
    "suggester_B_ref": "pseudonymous_id",
    "adjudicator_ref": "pseudonymous_id"
  }
}
```

---

## Correction Policy

**No edits. Ever.**

Found an error? Create a new packet with:
- `supersedes`: ID of old packet
- `correction_reason`: why

Old packet stays in database as historical record.

---

## Implementation (Simple)

Use **Git**:
- Main branch = canonical database
- Protected: require PR, signed commits
- CODEOWNERS: only transcribers merge to `/packets/`
- GitHub Actions: validate schema + hashes on every PR

No servers needed. Free. Immutable history built-in.

---

## Why This Matters

**For AI training:**
- Full lineage enables alignment audits
- Disagreement zones (A ≠ B) = high-information training signal

**For humans:**
- Audit trail survives decades
- Applicable to: legal records, scientific data, constitutional documents

**The insight:** Where A and B disagree reveals values, assumptions, hidden bias. That's the most valuable training data.
