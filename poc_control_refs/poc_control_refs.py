"""Proof of concept: compute `control` and `references` metadata for existing Pinecone chunks.

Reads chunks from Pinecone (read-only), applies a source profile (YAML) and optionally an LLM pass,
and writes a before/after comparison CSV plus a summary. Nothing is ever written back to Pinecone.

Run `python poc_control_refs.py --self-test` first; see README.md for the full usage.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "output"

# ---------------------------------------------------------------------------------------------------
# Read-only gate
# ---------------------------------------------------------------------------------------------------

ALLOWED_PINECONE_METHODS = ("list", "list_paginated", "fetch", "describe_index_stats")

# Any call that looks like a Pinecone write. The script refuses to run if its own source contains one.
WRITE_CALL_RE = re.compile(
    r"\.(upsert\w*|batch_upsert|update|delete\w*|create_index|configure_index|create_namespace|start_import)\s*\("
)


def check_source_is_read_only():
    """Static check: stop if this file contains a write-style call (see WRITE_CALL_RE)."""
    source = Path(__file__).read_text(encoding="utf-8")
    hits = []
    for line_no, line in enumerate(source.splitlines(), start=1):
        for m in WRITE_CALL_RE.finditer(line):
            hits.append(f"line {line_no}: {m.group(0)}")
    if hits:
        sys.exit("READ-ONLY CHECK FAILED, write-style calls found in source:\n  " + "\n  ".join(hits))


class ReadOnlyIndex:
    """Wraps a Pinecone index and exposes only the allowlisted read methods."""

    def __init__(self, real_index):
        # Name-mangled to _ReadOnlyIndex__index; no other code reads it.
        object.__setattr__(self, "_ReadOnlyIndex__index", real_index)

    def __getattr__(self, name):
        # Only called for attributes not found normally, i.e. everything except the private index.
        if name in ALLOWED_PINECONE_METHODS:
            return getattr(self.__index, name)
        raise PermissionError(f"Pinecone is read-only in this PoC: {name}")

    def __setattr__(self, name, value):
        raise PermissionError(f"Pinecone is read-only in this PoC: {name}")


def open_readonly_index():
    """The only function that touches the Pinecone SDK. Connects by host, never by index name."""
    api_key = os.environ.get("PINECONE_API_KEY")
    host = os.environ.get("PINECONE_HOST")
    if not api_key or not host:
        sys.exit("PINECONE_API_KEY and PINECONE_HOST must be set.")
    from pinecone import Pinecone

    return ReadOnlyIndex(Pinecone(api_key=api_key).Index(host=host))


# ---------------------------------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------------------------------

def load_profile(path):
    with open(path, encoding="utf-8") as f:
        profile = yaml.safe_load(f)
    for rule in profile["url_rules"]:
        rule["regex"] = re.compile(rule["pattern"], re.IGNORECASE)
    profile.setdefault("ignore_lines_containing", [])
    profile["reference_patterns"] = compile_reference_patterns(profile)
    return profile


def clean_text(text, profile):
    """Drop banner lines (e.g. the CDTFA disaster-relief banner) before extracting anything."""
    phrases = [p.lower() for p in profile["ignore_lines_containing"]]
    kept = [line for line in (text or "").splitlines() if not any(p in line.lower() for p in phrases)]
    return "\n".join(kept)


def as_text(value):
    """Metadata values can be strings or lists; turn either into one string."""
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return "" if value is None else str(value)


def normalize_family(tax_family):
    """'Sales_and_Use Tax' / ['sales and use tax'] -> 'sales and use tax'."""
    return re.sub(r"[_\-\s]+", " ", as_text(tax_family)).strip().lower()


def lookup_by_family(mapping, tax_family):
    """Find the mapping entry whose key appears in the chunk's tax_family; fall back to 'default'."""
    family = normalize_family(tax_family)
    for key, value in mapping.items():
        if key != "default" and key in family:
            return value
    return mapping.get("default")


# ---------------------------------------------------------------------------------------------------
# Step 2: control and doc_type
# ---------------------------------------------------------------------------------------------------

def fill_template(template, match):
    """'CA-RTC-{1}' + match -> 'CA-RTC-6011'. Keys are always upper-case (CA-ETUSA-6)."""
    return re.sub(r"\{(\d+)\}", lambda g: match.group(int(g.group(1))) or "", template).upper()


def match_url_rule(url, profile, start=0):
    """First matching rule wins. Returns (rule, match) or (None, None)."""
    for rule in profile["url_rules"][start:]:
        m = rule["regex"].search(url or "")
        if m:
            return rule, m
    return None, None


def number_regex(number):
    """Regex for a number where '.' and '-' are interchangeable, with digit boundaries on both sides.
    Letters glued on the right are fine ('325.0495STATE'), more digits are not ('60115', '6011.5')."""
    body = "".join(r"[.\-]" if ch in ".-" else re.escape(ch) for ch in number)
    return re.compile(rf"(?<!\d)(?<!\d[.\-]){body}(?!\d|[.\-]\d)")


def number_appears(number, meta, profile):
    haystack = "\n".join([
        as_text(meta.get("heading_path")),
        as_text(meta.get("title")),
        clean_text(as_text(meta.get("text")), profile)[:300],
    ])
    return bool(number_regex(number).search(haystack))


def compute_control(meta, profile):
    """Returns a dict with after_doc_type, after_control, control_reason and the rule details."""
    url = as_text(meta.get("url"))
    rule, m = match_url_rule(url, profile)
    result = {
        "profile_rule": "no_match", "rule_confidence": "", "is_compilation": "no",
        "after_doc_type": as_text(meta.get("document_type")), "after_control": "",
        "control_source": "", "control_reason": "no_rule_matched", "add_references": [],
    }
    if rule is None:
        return result

    result = {**result, "profile_rule": rule["id"], "rule_confidence": rule.get("confidence", ""),
              "after_doc_type": rule["doc_type"],
              "add_references": [fill_template(t, m) for t in rule.get("add_references", [])]}

    if rule.get("compilation"):
        result = {**result, "is_compilation": "yes", "control_reason": "compilation_page"}
        if rule["doc_type"] == "COMPILED":
            # "-all" pages: report the type of what they compile when the URL without "-all" matches a
            # later rule (e.g. vol2/suta/495-0000-all.html -> annotation_suta -> ANNOTATION).
            later_start = profile["url_rules"].index(rule) + 1
            inner_rule, _ = match_url_rule(re.sub(r"-all\.html$", ".html", url), profile, later_start)
            if inner_rule and not inner_rule.get("compilation"):
                result = {**result, "after_doc_type": inner_rule["doc_type"]}
        return result

    if rule.get("control_from_program"):
        folder = m.group(1)
        prefix = profile.get("program_codes", {}).get(folder)
        if not prefix:
            return {**result, "control_reason": f"unknown_program_folder:{folder}"}
        control = f"{prefix}-{m.group(2)}" + (f".{m.group(3)}" if m.group(3) else "")
    elif rule.get("control"):
        control = fill_template(rule["control"], m)
    else:
        return {**result, "control_reason": "rule_has_no_control"}

    # The number is the trailing numeric part of the key: CA-SUTA-195.0560 -> 195.0560.
    number = re.search(r"\d[\d.\-]*$", control).group(0)
    if not number_appears(number, meta, profile):
        return {**result, "control_reason": "url_number_not_in_text"}
    return {**result, "after_control": control, "control_source": "url_rule", "control_reason": "ok"}


# ---------------------------------------------------------------------------------------------------
# Step 3a: references by rules
# ---------------------------------------------------------------------------------------------------

SUB = r"\(\w{1,4}\)"
# Subdivisions after a number, dropped from the key: "6006(b)", "6011, subd. (c)(10)(D)", "subds. (a) and (b)".
SUBDIV = rf"(?:\s*,?\s*subds?\.\s*(?:{SUB})+(?:\s*(?:,|and|or)\s*(?:{SUB})+)*|(?:{SUB})+)?"
# Separators in a number list. Dashes and "through" mark a range.
SEP = r"(?:\s*,\s*(?:and\s+|or\s+|&\s*)?|\s+and\s+|\s+or\s+|\s*&\s*|\s*[-–]\s*|\s+through\s+)"
RANGE_SEP_RE = re.compile(r"[-–]|through", re.IGNORECASE)
# A bare "§" right after some other code's name ("Gov. Code, § 11342", "42 U.S.C. § 1983") is not ours.
OTHER_CODE_BEFORE_RE = re.compile(r"(?:Code|Regs|U\.?S\.?C|C\.?F\.?R|Act|tit\.\s*\d+)\.?\s*,?\s*$", re.IGNORECASE)


def compile_reference_patterns(profile):
    """One regex per reference_names entry: <name> <optional §/§§/Section> <number list>."""
    patterns = []
    for entry in profile["reference_names"]:
        num = entry["number"]
        item = rf"(?:§§?\s*)?(?:{num})(?!\d){SUBDIV}"
        names = "|".join(re.escape(n) for n in sorted(entry["names"], key=len, reverse=True))
        num_list = rf"{item}(?:{SEP}{item})*"
        full = rf"(?<![A-Za-z])(?:{names})(?![A-Za-z])\.?\s*,?\s*(?:§§?|sections?|secs?\.)?\s*(?P<list>{num_list})"
        patterns.append({
            "key_prefix": entry["key_prefix"],
            "num_list": num_list,
            "regex": re.compile(full, re.IGNORECASE),
            # Walks the number list one item at a time, keeping the separator before each item.
            "item_regex": re.compile(rf"(?P<sep>{SEP})?(?:§§?\s*)?(?P<num>{num})(?!\d){SUBDIV}", re.IGNORECASE),
        })
    return patterns


def numbers_in_list(item_regex, list_text):
    """'6001-6024, 6051' -> [('6001', True), ('6024', True), ('6051', False)]  (number, part of a range)."""
    items = []
    for m in item_regex.finditer(list_text):
        is_range = bool(m.group("sep") and RANGE_SEP_RE.search(m.group("sep")))
        if is_range and items:
            items[-1] = (items[-1][0], True)
        items.append((m.group("num"), is_range))
    return items


def resolve_prefix(key_prefix, tax_family, profile):
    if key_prefix == "ANNOTATION":
        return lookup_by_family(profile["annotation_prefix_by_tax_family"], tax_family)
    return key_prefix


def extract_refs_rules(text, tax_family, profile):
    """Returns a list of {key, span, method[, note]} found by the name patterns and bare § rule."""
    found = []
    taken_spans = []
    for pat in profile["reference_patterns"]:
        prefix = resolve_prefix(pat["key_prefix"], tax_family, profile)
        for m in pat["regex"].finditer(text):
            taken_spans.append(m.span())
            for number, is_range in numbers_in_list(pat["item_regex"], m.group("list")):
                ref = {"key": f"{prefix}-{number}".upper(), "span": m.group(0), "method": "rules"}
                found.append({**ref, "note": "range"} if is_range else ref)

    # Bare "§ 6011" / "Section 6051" with no code name: assume the family's default code (if any).
    bare_prefix = lookup_by_family({**profile.get("bare_section_default", {}), "default": None}, tax_family)
    bare_pat = next((p for p in profile["reference_patterns"] if p["key_prefix"] == bare_prefix), None)
    if bare_pat:
        bare_re = re.compile(rf"(?<![A-Za-z])(?:§§?|sections?)\s*(?P<list>{bare_pat['num_list']})", re.IGNORECASE)
        for m in bare_re.finditer(text):
            if any(s < m.end() and m.start() < e for s, e in taken_spans):
                continue  # already covered by a named match like "RTC § 6011"
            if OTHER_CODE_BEFORE_RE.search(text[max(0, m.start() - 40):m.start()]):
                continue
            for number, is_range in numbers_in_list(bare_pat["item_regex"], m.group("list")):
                ref = {"key": f"{bare_prefix}-{number}".upper(), "span": m.group(0),
                       "method": "bare_section_assumed"}
                found.append({**ref, "note": "range"} if is_range else ref)
    return found


# ---------------------------------------------------------------------------------------------------
# Step 3b: references by LLM
# ---------------------------------------------------------------------------------------------------

def allowed_prefixes(tax_family, profile):
    """{prefix: (names, number regex)} for this chunk, with ANNOTATION resolved to a real prefix."""
    return {resolve_prefix(e["key_prefix"], tax_family, profile): (e["names"], e["number"])
            for e in profile["reference_names"]}


def llm_prompt(text, tax_family, profile):
    prefixes = allowed_prefixes(tax_family, profile)
    prefix_lines = "\n".join(f"- {p}: written as {', '.join(names)}" for p, (names, _) in prefixes.items())
    bare = lookup_by_family({**profile.get("bare_section_default", {}), "default": None}, tax_family)
    bare_line = (f"A bare '§ 1234' or 'Section 1234' with no code name means {bare}." if bare
                 else "Skip a bare '§ 1234' with no code name.")
    banner = "; ".join(profile["ignore_lines_containing"]) or "(none)"
    return f"""You extract legal citations from a {profile['state']} tax document chunk.
Chunk tax_family: {as_text(tax_family)}

Allowed key prefixes (use only these):
{prefix_lines}

Rules:
- A key is <prefix>-<number>, e.g. CA-RTC-6011, CA-REG-1699.5.
- Drop subsections: "6006(b)" -> 6006, "1699.5(b)(4)" -> 1699.5.
- {bare_line}
- For a list like "§§ 6006, 6012" give one item per number. For a range "6001-6024" give the two endpoints.
- "span" must be copied exactly, character for character, from the text and must contain the number.
- Skip vague mentions with no number (e.g. "the Regulation").
- Skip anything in lines containing: {banner}
- Skip citations that do not fit an allowed prefix (federal law, other codes, case law).

Return JSON only: {{"references": [{{"span": "<exact text>", "key": "CA-RTC-6011"}}]}}

Text:
<<<
{text}
>>>"""


def validate_llm_items(items, text, tax_family, profile):
    """Keep only items with an allowed prefix, a span found in the text, and the key's number in the span."""
    prefixes = allowed_prefixes(tax_family, profile)
    kept, dropped = [], 0
    for item in items if isinstance(items, list) else []:
        key = str(item.get("key", "")).strip().upper() if isinstance(item, dict) else ""
        span = str(item.get("span", "")) if isinstance(item, dict) else ""
        prefix = max((p for p in prefixes if key.startswith(p + "-")), key=len, default=None)
        if not prefix or not span or span not in text:
            dropped += 1
            continue
        number = re.sub(r"\(.*$", "", key[len(prefix) + 1:])  # tolerate a left-over subsection
        if not re.fullmatch(prefixes[prefix][1], number, re.IGNORECASE) or not number_regex(number).search(span):
            dropped += 1
            continue
        kept.append({"key": f"{prefix}-{number}", "span": span, "method": "llm"})
    return kept, dropped


def load_llm_cache():
    path = OUTPUT_DIR / "llm_cache.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_llm_cache(cache):
    (OUTPUT_DIR / "llm_cache.json").write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")


def ask_llm(client, model, chunk_id, meta, text, profile, cache):
    """One call per chunk, cached by chunk ID + content_hash. Returns the raw list of items (or None on error)."""
    content_hash = as_text(meta.get("content_hash")) or hashlib.sha256(text.encode("utf-8")).hexdigest()
    cache_key = f"{chunk_id}|{content_hash}"
    if cache_key not in cache:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": llm_prompt(text, meta.get("tax_family"), profile)}],
                response_format={"type": "json_object"},
            )
            cache[cache_key] = json.loads(response.choices[0].message.content or "{}")
        except Exception as exc:  # report and carry on; the chunk just has no LLM references
            print(f"  LLM error for {chunk_id}: {type(exc).__name__}: {exc}")
            return None
        save_llm_cache(cache)
    return cache[cache_key].get("references", [])


# ---------------------------------------------------------------------------------------------------
# Step 1: scan and sample
# ---------------------------------------------------------------------------------------------------

def list_ids(index, namespace, scan):
    """List record IDs, stopping at `scan`. Handles both SDK shapes (lists of strings / .vectors[i].id)."""
    ids = []
    for page in index.list(namespace=namespace, limit=100):
        if hasattr(page, "vectors"):
            ids.extend(v.id for v in page.vectors)
        else:
            ids.extend(page)
        if len(ids) >= scan:
            break
    return ids[:scan]


def fetch_metadata(index, namespace, ids):
    """Fetch metadata in batches of 100. Returns {id: metadata dict}."""
    chunks = {}
    for start in range(0, len(ids), 100):
        response = index.fetch(ids=ids[start:start + 100], namespace=namespace)
        vectors = response.vectors if hasattr(response, "vectors") else response["vectors"]
        for vid, vec in vectors.items():
            meta = vec.metadata if hasattr(vec, "metadata") else vec.get("metadata")
            chunks[vid] = dict(meta or {})
        print(f"  fetched {min(start + 100, len(ids))}/{len(ids)}", end="\r")
    print()
    return chunks


def build_sample(groups, group_order, include_ids, limit, chunks):
    """--include-ids first, then round-robin across groups so every rule present is represented.
    Inside a group, also rotate across documents so one long document does not fill the group."""
    sample = [i for i in include_ids if i in chunks]
    queues = {}
    for group in group_order:
        by_doc = defaultdict(list)
        for cid in sorted(groups.get(group, [])):
            by_doc[as_text(chunks[cid].get("document_id")) or cid].append(cid)
        doc_lists = list(by_doc.values())
        interleaved = [lst[i] for i in range(max(map(len, doc_lists), default=0)) for lst in doc_lists if i < len(lst)]
        queues[group] = [c for c in interleaved if c not in sample]
    while len(sample) < limit and any(queues.values()):
        for group in group_order:
            if queues[group] and len(sample) < limit:
                sample.append(queues[group].pop(0))
    return sample


# ---------------------------------------------------------------------------------------------------
# Steps 4-6: resolve, CSV, summary
# ---------------------------------------------------------------------------------------------------

def link_for(url, anchor):
    """Our own url + heading_anchor; never invent an anchor."""
    anchor = as_text(anchor).strip()
    if not anchor:
        return url
    return url + (anchor if anchor.startswith("#") else "#" + anchor)


def build_lookup(chunks, controls):
    """control -> (chunk_id, url, heading_anchor), preferring the lowest chunk_index."""
    lookup = {}
    def position(cid):
        try:
            return int(chunks[cid].get("chunk_index", 0))
        except (TypeError, ValueError):
            return 0
    for cid in sorted(chunks, key=lambda c: (position(c), c)):
        control = controls[cid]["after_control"]
        if control and control not in lookup:
            lookup[control] = (cid, as_text(chunks[cid].get("url")), as_text(chunks[cid].get("heading_anchor")))
    return lookup


def csv_value(value):
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return "" if value is None else value


def write_csv(rows, orig_keys, path):
    columns = (["id"] + [f"orig_{k}" for k in orig_keys] + [
        "before_control", "after_control", "before_references", "after_references",
        "before_doc_type", "after_doc_type", "doc_type_changed",
        "profile_rule", "rule_confidence", "is_compilation", "control_source", "control_reason",
        "refs_rules", "refs_llm", "refs_only_rules", "refs_only_llm", "ref_details", "llm_dropped_count",
        "resolved_refs", "unresolved_refs", "resolved_links", "after_link",
    ])
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: csv_value(row.get(c)) for c in columns})


def pct(part, whole):
    return f"{(100 * part / whole):.1f}%" if whole else "n/a"


def build_summary(chunks, controls, groups, group_order, rows, use_llm, llm_errors, profile):
    lines = ["# Control / references PoC summary", ""]
    n = len(rows)
    sampled_per_group = Counter(r["profile_rule"] for r in rows)
    lines += [f"- Chunks scanned: {len(chunks)}", f"- Chunks sampled: {n}", f"- LLM pass: {'on' if use_llm else 'off (--no-llm)'}", "",
              "## Chunks per rule", "", "| rule | scanned | sampled |", "|---|---:|---:|"]
    lines += [f"| {g} | {len(groups.get(g, []))} | {sampled_per_group.get(g, 0)} |" for g in group_order]

    with_control = [r for r in rows if r["after_control"]]
    lines += ["", "## `control`", "", f"- Set on {len(with_control)} of {n} sampled chunks ({pct(len(with_control), n)})", "",
              "| control_reason | sampled chunks |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in Counter(r["control_reason"] for r in rows).most_common()]

    changed = [r["id"] for r in rows if r["doc_type_changed"] == "yes"]
    lines += ["", "## doc_type changes", "", f"- {len(changed)} sampled chunks where `after_doc_type` differs from stored `document_type`"]
    pairs = Counter(f"{r['before_doc_type'] or '(empty)'} -> {r['after_doc_type']}" for r in rows if r["doc_type_changed"] == "yes")
    lines += [f"  - {k}: {v}" for k, v in pairs.most_common()]
    lines += [f"- Examples: {', '.join(changed[:10])}" if changed else "- Examples: none"]

    rules_keys = sum(len(r["refs_rules"]) for r in rows)
    llm_keys = sum(len(r["refs_llm"]) for r in rows)
    total = sum(len(r["after_references"]) for r in rows)
    only_rules = sum(len(r["refs_only_rules"]) for r in rows)
    only_llm = sum(len(r["refs_only_llm"]) for r in rows)
    overlap = sum(len(set(r["refs_rules"]) & set(r["refs_llm"])) for r in rows)
    lines += ["", "## References (unique keys per chunk, summed over the sample)", "",
              f"- Total: {total}", f"- By rules: {rules_keys}", f"- By LLM: {llm_keys}", f"- Found by both: {overlap}",
              f"- Only rules: {only_rules}", f"- Only LLM: {only_llm}",
              f"- LLM items dropped by validation: {sum(r['llm_dropped_count'] or 0 for r in rows)}",
              f"- LLM calls that failed: {llm_errors}",
              f"- Chunks with at least one reference: {sum(1 for r in rows if r['after_references'])} of {n}"]

    resolved = sum(len(r["resolved_refs"]) for r in rows)
    unresolved = Counter(k for r in rows for k in r["unresolved_refs"])
    lines += ["", "## Reference resolution (against the scanned set)", "",
              f"- Resolved: {resolved} of {total} ({pct(resolved, total)})", "", "Top 20 unresolved keys:", "",
              "| key | count |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in unresolved.most_common(20)]

    inferred = [r["id"] for r in rows if r["rule_confidence"] == "INFERRED"]
    lines += ["", "## Chunks from INFERRED rules (review by hand)", "", ", ".join(inferred) if inferred else "None in the sample."]

    # Profile health, measured on the whole scan (not just the sample).
    lines += ["", "## Profile check against the scanned data", ""]
    unused = [r["id"] for r in profile["url_rules"] if not groups.get(r["id"])]
    lines.append(f"- URL rules that matched nothing in the scan: {', '.join(unused) or 'none'}")
    folders = Counter(c["control_reason"] for c in controls.values() if c["control_reason"].startswith("unknown_program_folder"))
    lines.append(f"- Statute folders without a program_codes entry: {dict(folders) or 'none'}")
    not_in_text = Counter(c["profile_rule"] for c in controls.values() if c["control_reason"] == "url_number_not_in_text")
    lines.append(f"- `url_number_not_in_text` in the scan, by rule: {dict(not_in_text) or 'none'}")
    lines.append("- `url_number_not_in_text` examples: " + ", ".join(
        f"{as_text(chunks[cid].get('url'))} (title {as_text(chunks[cid].get('title'))[:40]!r})"
        for cid in [c for c, info in controls.items() if info["control_reason"] == "url_number_not_in_text"][:5]))
    # tax_family is a list in the data; count each value on its own.
    families = Counter(normalize_family(v) for m in chunks.values()
                       for v in (m.get("tax_family") if isinstance(m.get("tax_family"), list) else [m.get("tax_family")]))
    known = [k for k in list(profile["annotation_prefix_by_tax_family"]) + list(profile.get("bare_section_default", {})) if k != "default"]
    lines.append("- tax_family values in the scan (chunks with each value; ✓ = matches a profile key):")
    lines += [f"  - {fam or '(empty)'!r}: {cnt} {'✓' if any(k in fam for k in known) else ''}" for fam, cnt in families.most_common()]
    no_match_prefixes = Counter(
        "/".join(re.sub(r"^https?://(www\.)?", "", as_text(chunks[c].get("url"))).split("/")[:3]) for c in groups.get("no_match", []))
    lines += ["- Most common URL prefixes with no matching rule:"]
    lines += [f"  - {p}: {cnt}" for p, cnt in no_match_prefixes.most_common(15)]
    type_changes = Counter(f"{c['profile_rule']}: {as_text(chunks[cid].get('document_type')) or '(empty)'} -> {c['after_doc_type']}"
                           for cid, c in controls.items()
                           if c["profile_rule"] != "no_match" and c["after_doc_type"] != as_text(chunks[cid].get("document_type")))
    lines += ["- doc_type changes in the scan, by rule:"]
    lines += [f"  - {k}: {v}" for k, v in type_changes.most_common(20)]
    return "\n".join(lines) + "\n"


def process_chunk(cid, meta, control_info, profile, lookup, llm):
    """Steps 3-4 for one sampled chunk. Returns the CSV row (lists kept as lists)."""
    text = clean_text(as_text(meta.get("text")), profile)
    tax_family = meta.get("tax_family")
    own = control_info["after_control"]

    rule_refs = extract_refs_rules(text, tax_family, profile)
    rule_refs += [{"key": k, "span": "", "method": "url_rule_add_references"} for k in control_info["add_references"]]
    llm_refs, dropped = [], ""
    if llm:
        raw = ask_llm(llm["client"], llm["model"], cid, meta, text, profile, llm["cache"])
        if raw is None:
            llm["errors"] += 1
        else:
            llm_refs, dropped = validate_llm_items(raw, text, tax_family, profile)

    rule_refs = [r for r in rule_refs if r["key"] != own]
    llm_refs = [r for r in llm_refs if r["key"] != own]
    keys_rules = sorted({r["key"] for r in rule_refs})
    keys_llm = sorted({r["key"] for r in llm_refs})
    after_refs = sorted(set(keys_rules) | set(keys_llm))
    details = []
    for r in rule_refs + llm_refs:
        if r not in details:
            details.append(r)

    resolved = [k for k in after_refs if k in lookup]
    before_doc_type = as_text(meta.get("document_type"))
    url = as_text(meta.get("url"))
    return {
        "id": cid,
        **{f"orig_{k}": v for k, v in meta.items()},
        "before_control": meta.get("control", ""),
        "after_control": own,
        "before_references": meta.get("references", ""),
        "after_references": after_refs,
        "before_doc_type": before_doc_type,
        "after_doc_type": control_info["after_doc_type"],
        "doc_type_changed": "yes" if control_info["after_doc_type"] != before_doc_type else "no",
        **{k: control_info[k] for k in ("profile_rule", "rule_confidence", "is_compilation", "control_source", "control_reason")},
        "refs_rules": keys_rules,
        "refs_llm": keys_llm,
        "refs_only_rules": sorted(set(keys_rules) - set(keys_llm)) if llm else [],
        "refs_only_llm": sorted(set(keys_llm) - set(keys_rules)),
        "ref_details": details,
        "llm_dropped_count": dropped,
        "resolved_refs": resolved,
        "unresolved_refs": [k for k in after_refs if k not in lookup],
        "resolved_links": [f"{k}={link_for(lookup[k][1], lookup[k][2])}" for k in resolved],
        "after_link": link_for(url, meta.get("heading_anchor")),
    }


# ---------------------------------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------------------------------

def self_test(profile):
    # --- Read-only gate, runtime part. The fake records calls so we can see what reached it.
    class FakeIndex:
        def __init__(self):
            self.calls = []
        def fetch(self, **kwargs):
            self.calls.append("fetch")
            return {"vectors": {}}
        def list(self, **kwargs):
            self.calls.append("list")
            return iter([["a#0"]])
        def upsert(self, **kwargs):  # must never be reachable through ReadOnlyIndex
            self.calls.append("upsert")

    fake = FakeIndex()
    ro = ReadOnlyIndex(fake)
    for blocked in ("upsert", "delete", "update"):
        try:
            getattr(ro, blocked)(ids=["x"])
            raise AssertionError(f"{blocked} was not blocked")
        except PermissionError as exc:
            assert str(exc) == f"Pinecone is read-only in this PoC: {blocked}", exc
    assert ro.fetch(ids=["a#0"], namespace="") == {"vectors": {}}
    assert list(ro.list(namespace="")) == [["a#0"]]
    assert fake.calls == ["fetch", "list"], fake.calls
    print("  ok  read-only gate (static check, upsert/delete/update blocked, fetch/list pass through)")

    # --- URL -> doc_type / control
    def control_for(url, text, title=""):
        return compute_control({"url": url, "text": text, "title": title, "tax_family": ["sales and use tax"]}, profile)

    cases = [
        ("https://cdtfa.ca.gov/lawguides/vol1/sutl/6011.html", "6011. Sales price.", "STATUTE_REFERENCE", "CA-RTC-6011"),
        ("https://cdtfa.ca.gov/lawguides/vol1/alt/additional-local-taxes-sec7286-50.html", "7286.50. Tax rate.",
         "STATUTE_REFERENCE", "CA-RTC-7286.50"),
        ("https://cdtfa.ca.gov/lawguides/vol2/suta/195-0560.html", "195.0560 Banks.", "ANNOTATION", "CA-SUTA-195.0560"),
        ("https://cdtfa.ca.gov/lawguides/vol2/suta/600-0000.html", "600.0000 Index", "OTHER", ""),
        ("https://cdtfa.ca.gov/lawguides/vol4/etusa/etusa-6.html", "6. Exemption.", "ANNOTATION", "CA-ETUSA-6"),
        ("https://www.boe.ca.gov/lawguides/property/current/ptlg/annt/220-0132.html", "220.0132 Assessment",
         "ANNOTATION", "CA-PTA-220.0132"),
        ("https://cdtfa.ca.gov/lawguides/annotations/320-0240.pdf", "Memorandum 320.0240", "LETTER_RULING", "CA-LTR-320.0240"),
    ]
    for url, text, doc_type, control in cases:
        got = control_for(url, text)
        assert (got["after_doc_type"], got["after_control"]) == (doc_type, control), (url, got)
    got = control_for("https://cdtfa.ca.gov/lawguides/vol2/suta/495-0000-all.html", "495.0000 all")
    assert got["is_compilation"] == "yes" and got["after_control"] == "" and got["control_reason"] == "compilation_page", got
    got = control_for("https://cdtfa.ca.gov/lawguides/annotations/320-0240.pdf", "Memorandum 320.0240")
    assert "CA-SUTA-320.0240" in got["add_references"], got
    got = control_for("https://cdtfa.ca.gov/taxes-and-fees/1507dp101819web.pdf", "Discussion paper")
    assert got["control_reason"] == "no_rule_matched" and got["after_control"] == "", got
    got = control_for("https://cdtfa.ca.gov/lawguides/vol1/sutl/6011.html", "Sales price means the total amount.")
    assert got["after_control"] == "" and got["control_reason"] == "url_number_not_in_text", got
    # Hard rule: '.' and '-' interchangeable, text glued on the right is fine.
    got = control_for("https://cdtfa.ca.gov/lawguides/vol2/suta/325-0495.html", "325-0495STATE OF CALIFORNIA")
    assert got["after_control"] == "CA-SUTA-325.0495", got
    print("  ok  URL -> doc_type / control")

    # --- Text -> references
    def keys(text, family="sales and use tax"):
        return sorted({r["key"] for r in extract_refs_rules(clean_text(text, profile), [family], profile)})

    assert keys("RTC §§ 6011") == ["CA-RTC-6011"]
    refs = extract_refs_rules("(§§ 6006(b), 6012, 6051.)", ["sales and use tax"], profile)
    assert sorted(r["key"] for r in refs) == ["CA-RTC-6006", "CA-RTC-6012", "CA-RTC-6051"], refs
    assert all(r["method"] == "bare_section_assumed" for r in refs), refs
    assert keys("Regulations 1546, 1699.5") == ["CA-REG-1546", "CA-REG-1699.5"]
    assert keys("Reg. 1699.5(b)(4)") == ["CA-REG-1699.5"]
    assert keys("Annotation 325.1480 (1/30/50)") == ["CA-SUTA-325.1480"]
    assert keys("Annotation 220.0132", "property tax") == ["CA-PTA-220.0132"]
    assert keys("Revenue and Taxation Code Section 6006(b)") == ["CA-RTC-6006"]
    assert keys("(Gov. Code, § 11342.) and 42 U.S.C. § 1983") == []
    assert keys("(Cal. Code Regs., tit. 18, § 1502, subd. (b).)") == ["CA-REG-1502"]
    assert keys("If you were affected by a California disaster, see Regulation 1234.\nOther text.") == []
    # Extra checks from the real data and the range rule.
    assert keys("(RTC §§ 6011, subd. (c)(10)(D), and 6012, subd. (c)(10)(D).)") == ["CA-RTC-6011", "CA-RTC-6012"]
    refs = extract_refs_rules("RTC §§ 6001-6024", ["sales and use tax"], profile)
    assert sorted(r["key"] for r in refs) == ["CA-RTC-6001", "CA-RTC-6024"] and all(r.get("note") == "range" for r in refs), refs
    print("  ok  text -> references")

    # --- LLM validation (no API call)
    text = "See RTC § 6011 and Regulation 1546(b)."
    items = [{"span": "RTC § 6011", "key": "CA-RTC-6011"},
             {"span": "Regulation 1546(b)", "key": "CA-REG-1546"},
             {"span": "not in text 1234", "key": "CA-REG-1234"},
             {"span": "RTC § 6011", "key": "CA-RTC-6012"},
             {"span": "RTC § 6011", "key": "US-IRC-6011"}]
    kept, dropped = validate_llm_items(items, text, ["sales and use tax"], profile)
    assert [k["key"] for k in kept] == ["CA-RTC-6011", "CA-REG-1546"] and dropped == 3, (kept, dropped)
    print("  ok  LLM output validation")

    assert link_for("https://x/a.html", "") == "https://x/a.html"
    assert link_for("https://x/a.html", "#sec") == "https://x/a.html#sec"
    print("SELF-TEST PASSED")


# ---------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", default=str(HERE / "ca_profile.yaml"))
    parser.add_argument("--limit", type=int, default=100, help="chunks in the final sample")
    parser.add_argument("--scan", type=int, default=5000, help="record IDs to scan")
    parser.add_argument("--include-ids", default="", help="comma-separated IDs that must be in the sample")
    parser.add_argument("--no-llm", action="store_true", help="skip the LLM pass")
    parser.add_argument("--self-test", action="store_true", help="run built-in checks and exit")
    args = parser.parse_args()

    check_source_is_read_only()
    profile = load_profile(args.profile)
    if args.self_test:
        self_test(profile)
        return

    print("PINECONE READ-ONLY MODE: allowed = " + ", ".join(ALLOWED_PINECONE_METHODS))
    namespace = os.environ.get("PINECONE_NAMESPACE", "")
    index = open_readonly_index()

    # Step 1: scan and sample
    include_ids = [i.strip() for i in args.include_ids.split(",") if i.strip()]
    print(f"Listing up to {args.scan} IDs in namespace {namespace!r} ...")
    ids = list_ids(index, namespace, args.scan)
    ids += [i for i in include_ids if i not in ids]
    print(f"Fetching metadata for {len(ids)} IDs ...")
    chunks = fetch_metadata(index, namespace, ids)
    missing = [i for i in include_ids if i not in chunks]
    if missing:
        print(f"WARNING: --include-ids not found in Pinecone: {missing}")

    # Step 2 runs on the whole scan: it is cheap and the lookup in step 4 needs it.
    controls = {cid: compute_control(meta, profile) for cid, meta in chunks.items()}
    groups = defaultdict(list)
    for cid, info in controls.items():
        groups[info["profile_rule"]].append(cid)
    group_order = [r["id"] for r in profile["url_rules"]] + ["no_match"]
    sample = build_sample(groups, group_order, include_ids, args.limit, chunks)
    order = {g: n for n, g in enumerate(group_order)}
    sample.sort(key=lambda c: (order[controls[c]["profile_rule"]], c))
    picked = Counter(controls[c]["profile_rule"] for c in sample)
    print("Scanned / picked per group:")
    for g in group_order:
        if groups.get(g):
            print(f"  {g:26s} {len(groups[g]):6d} / {picked.get(g, 0)}")

    # Steps 3-4
    lookup = build_lookup(chunks, controls)
    llm = None
    if not args.no_llm:
        from openai import OpenAI
        llm = {"client": OpenAI(), "model": os.environ.get("OPENAI_MODEL") or "gpt-5-mini",
               "cache": load_llm_cache(), "errors": 0}
        print(f"LLM pass with {llm['model']} (cache: {len(llm['cache'])} entries)")
    rows = []
    for n, cid in enumerate(sample, start=1):
        rows.append(process_chunk(cid, chunks[cid], controls[cid], profile, lookup, llm))
        print(f"  processed {n}/{len(sample)}", end="\r")
    print()

    # Steps 5-6
    orig_keys = sorted({k for c in sample for k in chunks[c]}, key=lambda k: (k == "text", k))
    write_csv(rows, orig_keys, OUTPUT_DIR / "control_refs_poc.csv")
    summary = build_summary(chunks, controls, groups, group_order, rows, llm is not None,
                            llm["errors"] if llm else 0, profile)
    (OUTPUT_DIR / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Wrote {OUTPUT_DIR / 'control_refs_poc.csv'} and {OUTPUT_DIR / 'summary.md'}")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    main()
