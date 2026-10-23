# `control` / `references` proof of concept

Reads existing chunks from Pinecone (**read-only**), computes two candidate metadata fields and writes a
before/after CSV so you can judge them before changing the ingestion pipeline.

- **`control`**: the one legal authority a chunk *is*, as a key (e.g. `CA-RTC-6011` for
  `cdtfa.ca.gov/lawguides/vol1/sutl/6011.html`). It comes from the URL pattern and is kept only if the URL's
  number also appears in the chunk's `heading_path`, `title` or first 300 characters of `text`.
- **`references`**: a flat list of keys for authorities the chunk text *mentions*
  (e.g. `["CA-REG-1546", "CA-RTC-6011"]`), found by regex rules and, optionally, by an LLM.

A reference resolves when some scanned chunk has that key as its `control`. The link is always built from our
own metadata: `url` + `heading_anchor` (bare `url` when there is no anchor).

## Install and run

```bash
cd poc_control_refs
pip install -r requirements.txt

export PINECONE_API_KEY=...        # read-only key, role DataPlaneViewer
export PINECONE_HOST=...           # index host URL (the index is never looked up by name)
export PINECONE_NAMESPACE=...      # default ""
export OPENAI_API_KEY=...          # only needed without --no-llm
export OPENAI_MODEL=gpt-5-mini     # default gpt-5-mini

python poc_control_refs.py --self-test           # offline checks, no Pinecone or OpenAI calls

# Free first run: rules only, no LLM
python poc_control_refs.py --profile ca_profile.yaml --limit 100 --scan 5000 \
  --include-ids b05eb9b0d23e93be#2 --no-llm

# With the LLM pass (responses cached in output/llm_cache.json, so reruns cost nothing)
python poc_control_refs.py --profile ca_profile.yaml --limit 100 --scan 5000 \
  --include-ids b05eb9b0d23e93be#2
```

| Option | Meaning |
|---|---|
| `--profile` | source profile YAML (default `ca_profile.yaml`) |
| `--limit` | chunks in the final sample (default 100) |
| `--scan` | record IDs to scan for the sample and the control lookup table (default 5000) |
| `--include-ids` | comma-separated IDs that must be in the sample |
| `--no-llm` | skip the LLM pass |
| `--self-test` | run the built-in checks and exit |

Outputs in `output/`: `control_refs_poc.csv`, `summary.md` (also printed), `llm_cache.json`.

## Read-only gate

1. **Key**: use a Pinecone key whose only role is `DataPlaneViewer`; Pinecone rejects any write.
2. **Wrapper**: `open_readonly_index()` is the only code that touches the Pinecone SDK. It returns a
   `ReadOnlyIndex` that passes through only `list`, `list_paginated`, `fetch` and `describe_index_stats` and raises
   `PermissionError("Pinecone is read-only in this PoC: <name>")` for anything else.
3. **Checks**: on every start the script scans its own source for write-style calls (upsert, update, delete,
   create/configure index, create namespace, start import) and refuses to run if it finds one; `--self-test`
   also checks that the wrapper blocks writes against a stub index. The banner
   `PINECONE READ-ONLY MODE: allowed = ...` is printed before any Pinecone call.

## CSV columns (`output/control_refs_poc.csv`)

One row per sampled chunk. UTF-8 with BOM so Excel opens it cleanly. Lists are JSON strings.

| Column | Meaning |
|---|---|
| `id` | Pinecone record ID (`document_id#chunk position`) |
| `orig_<key>` | every original metadata field, unchanged (`orig_text` is the chunk text) |
| `before_control` | stored `control` value (empty today) |
| `after_control` | computed `control` key, empty when not safe to set |
| `before_references` | stored `references` value |
| `after_references` | computed reference keys: sorted union of rules + LLM |
| `before_doc_type` | stored `document_type` |
| `after_doc_type` | `doc_type` from the matching URL rule (stored value when no rule matched) |
| `doc_type_changed` | `yes` if `after_doc_type` differs from `before_doc_type` |
| `profile_rule` | id of the first matching URL rule, or `no_match` |
| `rule_confidence` | `VERIFIED` or `INFERRED` (inferred rules need a manual check) |
| `is_compilation` | `yes` for pages holding many authorities (`-all` pages, court-decision digests) |
| `control_source` | `url_rule` when `after_control` is set, else empty |
| `control_reason` | `ok`, `no_rule_matched`, `rule_has_no_control`, `compilation_page`, `unknown_program_folder:<x>`, `url_number_not_in_text` |
| `refs_rules` | keys found by the regex rules (including the rule's `add_references`) |
| `refs_llm` | keys found by the LLM that passed validation |
| `refs_only_rules` | keys only the rules found (empty with `--no-llm`) |
| `refs_only_llm` | keys only the LLM found |
| `ref_details` | JSON list of `{key, span, method}` (+ `note: "range"` for range endpoints). `method` is `rules`, `bare_section_assumed` (bare `§` with the family's default code), `url_rule_add_references` or `llm` |
| `llm_dropped_count` | LLM items rejected by validation (bad prefix, span not in text, number not in span) |
| `resolved_refs` | references that match the `control` of some scanned chunk |
| `unresolved_refs` | references with no matching chunk in the scanned set |
| `resolved_links` | `key=url#anchor` for each resolved reference (no `#anchor` if the target has none) |
| `after_link` | this chunk's own link: `url` + `heading_anchor`, or bare `url` |

## Adding a state or a rule (YAML only)

Copy `ca_profile.yaml` (e.g. to `tx_profile.yaml`) and edit only the YAML:

- `url_rules`: checked in order, first match wins, so put specific rules (index pages, compilations) first.
  `pattern` is a regex searched in the full URL; `{1}`, `{2}` in `control` / `add_references` are its capture
  groups. Use `control_from_program: true` with `program_codes` when a folder name decides the code,
  `compilation: true` for many-authority pages, and `confidence: INFERRED` for unchecked patterns.
- `reference_names`: for each key prefix, how the code is written in text (`names`) and what its numbers look
  like (`number` regex). Add a spelling by adding it to `names`.
- `annotation_prefix_by_tax_family`, `bare_section_default`: choose a prefix from the chunk's `tax_family`
  (a key matches when it appears in the tax_family value).
- `ignore_lines_containing`: lines with these phrases (site banners) are dropped before extraction.

Then run with `--profile tx_profile.yaml`.

## Known PoC limits

- No writes to Pinecone, ever: results exist only in the CSV.
- References resolve only within the scanned set (`--scan` IDs, listed in ID order), so the resolution rate
  is a lower bound, and keys whose target type has no URL rule (e.g. regulations) can never resolve yet.
- Ranges ("6001-6024") are stored as their two endpoints only.
- Subsections are dropped from keys (`6006(b)` -> `CA-RTC-6006`).
- A bare `§` with no code name is assumed to be the family's default code (`bare_section_assumed`), unless it
  directly follows another code's name (e.g. "Gov. Code, §").
