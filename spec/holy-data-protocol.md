# Holy Data Protocol (HDP) v0.1

**A bias-resistant text lineage and adjudication protocol with immutable audit trails.**

## Purpose

Create training and archival text data with full traceability from original source through neutral rewrite, blind suggestions, adjudication, and final accepted text. Preserve history without erasure by storing every decision as an append-only record.

## Core Insight

Most systems fail because they allow feedback loops too early. HDP explicitly blocks:
- Groupthink
- Status influence
- Rhetorical dominance
- Early consensus bias

This is closer to Byzantine fault tolerance for meaning, or air-gapped epistemology.

---

## Roles and Separation Rules

### Roles

| Role | Function | Constraints |
|------|----------|-------------|
| **Author** | Creates original material | Never touches the database |
| **Normalizer (Team 1)** | Rewrites for neutrality using dictionary/thesaurus | Cannot see suggestions or final decisions |
| **Suggester A** | Proposes diffs independently | Cannot communicate with B or see B's work |
| **Suggester B** | Proposes diffs independently | Cannot communicate with A or see A's work |
| **Adjudicator (Team 2)** | Reviews all inputs, votes on changes | Cannot edit canonical database |
| **Transcriber** | Enters closed packets exactly as written | Cannot modify content, only transcribe |
| **Auditor** (optional) | Verifies integrity via hashes and procedure compliance | Read-only access |

### Hard Separation Requirements

- No single human may both **influence content** and **commit canonical records**
- Transcribers never propose changes
- Suggesters never communicate with each other
- Normalizers never see suggester output or adjudication outcomes until packet closure
- Adjudicators never edit the canonical database directly

---

## Data Model

### Unit of Work

A **Segment** is the atomic unit—a sentence, numbered clause, or paragraph. Pick one unit size and maintain consistency across a dataset.

### Segment Identifiers

```
dataset_id / document_id / segment_id / segment_order
```

`segment_id` must be stable and never reused.

### Reason Codes

Structured tags applied to each adjudication decision. Multiple codes allowed per decision.

**Core Codes:**
- `clarity` — Improves readability or reduces ambiguity
- `factual_correction` — Fixes factual errors
- `bias_reduction` — Reduces framing, emotional language, or hidden assumptions
- `redundancy` — Removes unnecessary repetition
- `intent_preservation` — Maintains original meaning while improving form

**Optional Extension Codes:**
- `safety`
- `formatting`
- `style_consistency`
- `scope_control`
- `ambiguity_resolution`

**Rule:** Keep codes short and controlled. Avoid free-text explanations except in rare edge cases.

---

## Workflow Phases

### Phase 0: Authoring
- **Input:** Raw source material
- **Output:** `original_text` (preserved exactly as received)
- **Constraint:** No normalization, no edits

### Phase 1: Normalization
- **Input:** `original_text`
- **Output:** `normalized_text_v1`

**Normalizer Rules:**
- Preserve meaning and intent class
- No new claims
- No persuasion
- Use neutral tone
- Dictionary and thesaurus assistance allowed
- If factual statement is ambiguous, rewrite for clarity without changing claim content

**Required Metadata:**
- `intent_class`: one of `informative`, `persuasive`, `speculative`, `narrative`
- `normalization_ruleset_version`

**Critical:** Normalization must not change intent class. Only wording may change.

### Phase 2: Blind Suggestions
- **Input:** `normalized_text_v1`
- **Output:** `suggestion_A_diffs`, `suggestion_B_diffs`

**Constraints:**
- Suggesters submit diffs only, not full rewrites
- No awareness of the other suggester's work
- Each diff anchored to specific spans in `normalized_text_v1`

### Phase 3: Adjudication
- **Input:** `original_text`, `normalized_text_v1`, `suggestion_A_diffs`, `suggestion_B_diffs`
- **Output:** `final_text`, `decision_log`, `reason_codes`

**Decision Process:**
- Team 2 evaluates each proposed change item
- Vote outcome per item: `accept`, `reject`, `accept_with_modification`
- Majority vote required; ties resolved by predefined rule (chair vote or default reject)

**Required Per Decision Item:**
- `decision_item_id`
- `target_span` (reference to normalized_text_v1 offsets)
- `proposal_source`: `A`, `B`, `both`, or `Team2_mod`
- `action`: `accept`, `reject`, `accept_with_modification`
- `reason_codes`: list from controlled vocabulary
- `vote_tally`: counts

**High-Information Zones:** Flag items where A and B disagree sharply. These are prime training signal—AI learns more from fractures than harmony.

### Phase 4: Transcription and Commit
- **Input:** Closed packet from adjudication
- **Output:** Canonical database entry

**Constraints:**
- Transcribers enter packet fields exactly as written
- No content changes permitted at commit time
- Commit is append-only; no edits; corrections create new packets

**Verification:**
- Double entry by two independent transcribers
- System compares canonical hashes before accepting commit
- Mismatch = rejection, manual review

---

## Closed Segment Packet Specification

A **Closed Segment Packet** is the only artifact that gets committed to the canonical database.

### Required Fields

#### 1. Identity
```json
{
  "dataset_id": "string",
  "document_id": "string",
  "segment_id": "string",
  "segment_order": "integer",
  "created_utc": "ISO8601",
  "closed_utc": "ISO8601"
}
```

#### 2. Text Lineage
```json
{
  "original_text": "string",
  "normalized_text_v1": "string",
  "final_text": "string"
}
```

#### 3. Suggestions
```json
{
  "suggestion_A_diffs": [/* diff items */],
  "suggestion_B_diffs": [/* diff items */]
}
```

**Diff Item Format:**
```json
{
  "diff_id": "string",
  "op": "insert | delete | replace",
  "start_offset": "integer",
  "end_offset": "integer",
  "anchor_before": "string (short context)",
  "anchor_after": "string (short context)",
  "new_text": "string (for insert/replace)",
  "proposer_id": "A | B"
}
```

#### 4. Adjudication Record
```json
{
  "decision_log": [/* decision items */],
  "intent_class": "informative | persuasive | speculative | narrative",
  "normalization_ruleset_version": "string",
  "adjudication_ruleset_version": "string"
}
```

**Decision Item Format:**
```json
{
  "decision_item_id": "string",
  "target_span": {"start": "int", "end": "int"},
  "proposal_source": "A | B | both | Team2_mod",
  "action": "accept | reject | accept_with_modification",
  "reason_codes": ["clarity", "bias_reduction"],
  "vote_tally": {"accept": 2, "reject": 1}
}
```

#### 5. Integrity
```json
{
  "hash_original": "SHA256",
  "hash_normalized": "SHA256",
  "hash_suggestion_A": "SHA256",
  "hash_suggestion_B": "SHA256",
  "hash_final": "SHA256",
  "hash_packet": "SHA256"
}
```

**Hash Requirements:**
- Use SHA-256 consistently across the system
- Compute hashes over canonical serialized forms (UTF-8, LF line endings, trimmed trailing spaces)
- `hash_packet` covers the entire packet minus the hash_packet field itself

#### 6. Provenance
```json
{
  "author_ref": "pseudonymous_id",
  "normalizer_team_ref": "pseudonymous_id",
  "suggester_A_ref": "pseudonymous_id",
  "suggester_B_ref": "pseudonymous_id",
  "adjudicator_team_ref": "pseudonymous_id",
  "transcriber_1_ref": "pseudonymous_id",
  "transcriber_2_ref": "pseudonymous_id"
}
```

---

## Append-Only Correction Policy

**No record is ever edited.**

If an error is found:
1. Create a **Correction Packet** that references the prior `packet_id`
2. Provide corrected fields
3. Provide `correction_reason`: `transcription_error`, `ruleset_bug`, `discovered_fact_error`, `other`
4. Recompute hashes for the new packet
5. Link old and new via `supersedes` pointer

The old packet remains in the database as historical truth of what was recorded at the time.

---

## Implementation Options

### Minimal Stack (Recommended Start)

**Git + Signed Commits + JSONL**

- Source of truth: Git repository
- Canonical format: JSON Lines (one packet per line)
- Signed commits for tamper evidence
- Pre-commit hooks for schema validation
- Protected main branch (only transcribers can merge)

### Scaling Up

**Event Sourcing in Postgres**

- `segment_packets` table for closed packets
- `events` table for every phase event
- Row-level security for role separation
- Daily hash chain for tamper detection

### Maximum Tamper Resistance

**Git + WORM Storage + Public Hash Chain**

- Closed packets committed to Git
- Daily export to WORM (Write-Once-Read-Many) storage
- Publish daily root hash publicly

---

## GitHub Implementation

### Directory Structure
```
/segments/{dataset_id}/{document_id}/segment_{id}.json
/suggestions/{dataset_id}/{document_id}/suggestion_A_{id}.json
/suggestions/{dataset_id}/{document_id}/suggestion_B_{id}.json
/packets/{dataset_id}/{document_id}/packet_{id}.json
```

### Branch Protection
- `main` branch: require PR, require approvals, require signed commits, no force pushes
- Only transcriber team can merge to `main`

### CODEOWNERS
```
/packets/ @transcribers-team
/schemas/ @core-admins
```

### Issue Templates

**Suggestion A Template:**
- segment_id
- normalized_text_hash
- diff_operation
- start_offset / end_offset
- anchor_before / anchor_after
- proposed_text
- justification (optional, short)
- Auto-label: `suggestion-A`

**Suggestion B Template:** Same structure, auto-label: `suggestion-B`

---

## Validation Rules

A packet is valid only if:

1. All required fields exist
2. Offsets match `normalized_text_v1` bounds
3. `final_text` equals result of applying accepted diffs plus Team2 modifications
4. All hashes match recomputation
5. Double-entry hashes match each other
6. Role separation constraints satisfied by user IDs

---

## Why This Protocol Matters

### For AI Training
- Full lineage enables alignment audits
- Disagreement zones (A ≠ B) are high-information training signal
- Provenance control supports developmental alignment paradigm

### For Human Knowledge
- Audit trail that survives decades
- History never disappears, only accumulates
- Applicable to: scientific datasets, legal records, constitutional documents, cultural archives

### For Governance
- Separation of powers applied to language
- Bias-resistant consensus formation
- AI is the beneficiary; humans are the authors

---

## Version History

- **v0.1** — Initial specification (2025)
