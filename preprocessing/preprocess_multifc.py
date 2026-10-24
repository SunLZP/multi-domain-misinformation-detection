"""Preprocess MultiFC claims; audit native ratings and export source-aware binary targets. Python 3.10+; no external packages."""

import argparse
import uuid
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import re
import sys
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET = 'multifc'

META = ["claim_url", "reason", "category", "speaker", "checker", "tags",
        "article_title", "publication_date", "claim_date", "entities"]
FIELDS = ["id", "split", "label", "domain", "group_id", "text", "text_raw",
          "empty_text", "replacement_characters", "word_count", "source_file"] + META

def find_input(filename):
    """Support scripts beside ZIPs or inside a preprocessing subfolder."""
    candidates = [ROOT / filename, ROOT / "data" / filename,
                  ROOT.parent / filename, ROOT.parent / "data" / filename]
    matches = [candidate.resolve() for candidate in candidates if candidate.is_file()]
    if not matches:
        searched = "\n".join(f"  {candidate}" for candidate in candidates)
        raise FileNotFoundError(f"Cannot find {filename}. Searched:\n{searched}")
    # Prefer the ZIP beside the script, then local data/, then the parent.
    # A CLI --input remains available when intentionally choosing another release.
    return matches[0]


# Project-defined binary harmonisation v1, not the original publishers' binary labels.
# Majority-veracity policy: mostly true/correct -> 0; mostly false/fiction -> 1.
# Mixed, uncertain and non-veracity labels are excluded, never guessed.
# GossipCop is a 0..10 scale: original 0 -> fake(1), original 10 -> real(0).
BINARY_RULES = {'abbc': {0: [], 1: []},
 'afck': {0: ['correct', 'mostly-correct'], 1: ['incorrect']},
 'bove': {0: [], 1: ['rating: false']},
 'chct': {0: ['verdict: true'], 1: ['verdict: false']},
 'clck': {0: [], 1: ['incorrect']},
 'faan': {0: ['factscan score: true'], 1: ['factscan score: false']},
 'faly': {0: ['true'], 1: ['false']},
 'fani': {0: ['conclusion: accurate'], 1: ['conclusion: false']},
 'farg': {0: [], 1: ['false']},
 'goop': {0: ['10'], 1: ['0']},
 'hoer': {0: ['true messages'], 1: ['bogus warning', 'fake news']},
 'huca': {0: [], 1: []},
 'mpws': {0: ['accurate'], 1: ['false']},
 'obry': {0: ['verified', 'mostly_true'], 1: ['mostly_false']},
 'para': {0: ['true', 'mostly true'], 1: ['false', 'mostly false', 'pants on fire!']},
 'peck': {0: ['true'], 1: ['false']},
 'pomt': {0: ['true', 'mostly true'], 1: ['false', 'mostly false', 'pants on fire!']},
 'pose': {0: [], 1: []},
 'ranz': {0: ['fact'], 1: ['fiction']},
 'snes': {0: ['true', 'mostly true'], 1: ['false', 'mostly false']},
 'thal': {0: ['mostly true'], 1: ['mostly false', 'we rate this claim false']},
 'thet': {0: ['true', 'mostly true'], 1: ['false', 'mostly false']},
 'tron': {0: ['truth!', 'mostly truth!'], 1: ['fiction!', 'mostly fiction!']},
 'vees': {0: [], 1: ['fake', 'false']},
 'vogo': {0: ['determination: true', 'determination: mostly true'],
          1: ['determination: false', 'determination: huckster propaganda']},
 'wast': {0: [], 1: ['false']}}
BINARY_SOURCES = {'africa_check': 'https://africacheck.org/how-we-fact-check/how-we-rate-claims',
 'binary_grouping': 'https://misinforeview.hks.harvard.edu/article/fact-checking-fact-checkers-a-data-driven-approach/',
 'check_your_fact': 'https://checkyourfact.com/methodology/',
 'dataset': 'https://aclanthology.org/D19-1475.pdf',
 'factcheckni': 'https://factcheckni.org/articles/news/for-your-consideration-factcheckni-verdict-ratings/',
 'gossipcop_scale': 'https://www.cs.emory.edu/~kshu5/files/FakeNewsNet_big_data.pdf',
 'politifact': 'https://www.politifact.com/article/2018/feb/12/principles-truth-o-meter-politifacts-methodology-i/',
 'truth_or_fiction': 'https://www.truthorfiction.com/fact-check-ratings/',
 'voice_of_san_diego': 'https://voiceofsandiego.org/category/fact/'}
BINARY_FIELDS = ["id", "text", "label", "label_id", "original_label", "domain", "group_id", "split"]
BINARY_POLICY = "source-aware-majority-veracity-v1"


def binary_decision(domain, label):
    source_rules = BINARY_RULES.get(domain, {})
    for target, allowed in source_rules.items():
        if label in allowed:
            return target, "explicit_source_rule"
    if domain == "pose" or label in {"full flop", "half flip", "no flip"}:
        return None, "non_veracity_promise_or_position"
    if label in {"half-true", "half true", "partly true", "partially true", "mixture", "in-between"} or " & " in label:
        return None, "mixed_or_partial_truth"
    if label in {"none", "unproven", "unproven!", "unverified", "unobservable", "no evidence", "unsupported", "disputed!", "investigation pending!", "verdict: unsubstantiated", "conclusion: unclear", "unsubstantiated messages"}:
        return None, "unknown_or_insufficient_evidence"
    if "outdated" in label or label == "previously truth! now resolved!":
        return None, "time_dependent_rating"
    return None, "outside_reviewed_binary_rules"


def binary_record(row):
    target, reason = binary_decision(row["domain"], row["label"])
    if target is None:
        raise ValueError(f"Excluded rating passed to binary export: {row['domain']} / {row['label']}")
    return {"id": row["id"], "text": row["text"], "label": target, "label_id": target,
            "original_label": row["label"], "domain": row["domain"],
            "group_id": row["group_id"], "split": row["split"]}


def clean(text):
    # Preserve case, punctuation, numbers, stopwords and negation.
    return " ".join(unicodedata.normalize("NFC", text).split())


def key(text):
    return clean(text).casefold()


def checked_zip(source):
    archive = zipfile.ZipFile(source, mode="r")
    bad = archive.testzip()
    if bad:
        archive.close()
        raise ValueError(f"ZIP CRC failure: {bad}")
    if len(archive.namelist()) != len(set(archive.namelist())):
        archive.close()
        raise ValueError("Duplicate ZIP entry names")
    return archive


def load_multifc(path):
    rows = []
    with checked_zip(path) as outer:
        with checked_zip(io.BytesIO(outer.read("reference_multi_fc.zip"))) as reference:
            solutions = reference.read("test.solution").decode("utf-8").splitlines()
            known_labels = set(reference.read("labels.txt").decode("utf-8").splitlines())
        with checked_zip(io.BytesIO(outer.read("public_data.zip"))) as public:
            for split in ("train", "dev", "test"):
                name = f"{split}.tsv"
                # LF records only. U+2028 within fields is not a record boundary.
                text = public.read(name).decode("utf-8")
                records = text.split("\n")
                if records[-1] == "":
                    records.pop()
                if split == "test" and len(records) != len(solutions):
                    raise ValueError("Test/solution length mismatch; cannot align labels")
                for index, record in enumerate(records):
                    cells = record.removesuffix("\r").split("\t")
                    expected = 12 if split == "test" else 13
                    if len(cells) != expected:
                        raise ValueError(f"{name} record {index + 1}: expected {expected} fields, got {len(cells)}")
                    if split == "test":
                        cells.insert(2, solutions[index])
                    identifier, raw, label, *metadata = cells
                    if not re.fullmatch(r"[a-z]{4}-\d+", identifier):
                        raise ValueError(f"Unexpected claim ID: {identifier}")
                    if label not in known_labels:
                        raise ValueError(f"Unknown label: {label}")
                    rows.append(dict(dataset="MultiFC", id=identifier, split=split,
                                     label=label, domain=identifier.split("-")[0],
                                     text_raw=raw, text=clean(raw), source_file=name,
                                     **dict(zip(META, metadata))))
    return rows


def check(condition, message):
    if not condition:
        raise ValueError(message)
    print(f"[PASS] {message}")


def csv_roundtrip(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        check(reader.fieldnames == fields, f"{path.name}: CSV column names verified")
        saved = list(reader)
    expected = [{k: str(row.get(k, "")) for k in fields} for row in rows]
    check(saved == expected, f"{path.name}: all {len(rows):,} rows match after CSV readback")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    try:
        main()
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, RuntimeError) as error:
        print(f"[FAIL] {error}", file=sys.stderr)
        print("Processing stopped. Fix the problem above and rerun; existing outputs may be from an earlier run.", file=sys.stderr)
        sys.exit(1)


def fallback_output(dataset):
    # Stable, project-specific results path; this never relocates code or datasets.
    project_key = hashlib.sha256(str(ROOT).encode("utf-8")).hexdigest()[:8]
    return Path.home() / "EE6405-results" / f"{ROOT.name}-{project_key}" / dataset


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with path.open(encoding="utf-8") as handle:
        restored = [json.loads(line) for line in handle]
    check(restored == rows, f"{path.name}: JSONL content and row count verified")


def verify_manifest(path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") not in (2, 3) or not manifest.get("complete"):
        raise ValueError(f"Incomplete or incompatible manifest: {path}")
    for name, expected in manifest["files"].items():
        target = (path.parent / name).resolve()
        if not target.is_relative_to(path.parent.resolve()):
            raise ValueError("Manifest contains an invalid file path")
        if sha256(target) != expected["sha256"]:
            raise ValueError(f"Generated file changed or incomplete: {target}")
    return manifest


def load_preprocessed(profile="binary", output_dir=None):
    """Next-stage API: return {split: records} after verifying completed output hashes.

    Example: import preprocess_multifc; data = preprocess_multifc.load_preprocessed()
    Only paths and text/labels are loaded; no training is performed.
    """
    candidates = [Path(output_dir)] if output_dir else [ROOT / "output" / DATASET, fallback_output(DATASET)]
    valid = []
    for directory in candidates:
        try:
            manifest = verify_manifest(directory / "manifest.json")
            valid.append((manifest["created_utc"], directory, manifest))
        except (OSError, ValueError, KeyError):
            continue
    if not valid:
        raise ValueError("No verified completed output. Run this preprocessing script first.")
    _, directory, manifest = max(valid, key=lambda item: item[0])
    if profile not in manifest["profiles"]:
        raise ValueError(f"Unavailable profile {profile!r}; rerun preprocessing to generate it. Available: {list(manifest['profiles'])}")
    result = {}
    for split, relative in manifest["profiles"][profile].items():
        with (directory / relative).open(encoding="utf-8") as handle:
            result[split] = [json.loads(line) for line in handle]
    return result


def publish_outputs(requested, producer, source):
    """Probe actual writes, recover once outside the project, commit manifest last."""
    failures = []
    destinations = [requested.resolve(), fallback_output(DATASET).resolve()]
    for output in dict.fromkeys(destinations):
        # A source archive may never be inside a destination we publish into.
        if source.resolve().is_relative_to(output):
            raise ValueError("Output folder must not contain the source ZIP; choose a separate results folder")
        try:
            output.mkdir(parents=True, exist_ok=True)
            probe = output / f".write-test-{uuid.uuid4().hex}"
            try:
                probe.write_text("write check", encoding="utf-8")
            finally:
                if probe.exists():
                    probe.unlink()
            # Invalidate any earlier success marker before replacing output files.
            marker = output / "manifest.json"
            if marker.exists():
                marker.unlink()
            manifest = producer(output)
            manifest.update(schema_version=3, complete=True,
                            created_utc=datetime.now(timezone.utc).isoformat(),
                            output_directory=str(output), project_directory=str(ROOT),
                            script_sha256=sha256(Path(__file__)))
            write_json(marker, manifest)
            verify_manifest(marker)
            print(f"[PASS] Output files, hashes and completed manifest verified")
            print(f"[DONE] Results: {output}")
            print(f"[NEXT] Import {Path(__file__).stem} and call load_preprocessed(). See README.md.")
            return output
        except OSError as error:
            failures.append(f"{output}: {error}")
            print(f"[WARN] Cannot save results to {output}: {error}")
            print("[INFO] Trying the automatic results folder; originals stay in place.")
    raise OSError("No writable output location.\n" + "\n".join(failures))


def file_manifest(output, names):
    return {name: {"sha256": sha256(output / name), "bytes": (output / name).stat().st_size}
            for name in names}


def model_record(row):
    # Native-label audit profile; the default loader uses binary exports instead.
    result = {field: row[field] for field in ["id", "text", "label", "domain", "group_id", "split"]}
    result["label_key"] = row["domain"] + "::" + row["label"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Optional ZIP override; otherwise find it beside the script or one folder above")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "multifc")
    args = parser.parse_args()
    args.input = args.input if args.input is not None else find_input('MultiFC_Codalab.zip')
    print("MultiFC: claim preprocessing")
    print(f"Input: {args.input.resolve()}")
    check(args.input.is_file(), "Input ZIP exists")
    before = sha256(args.input)
    rows = load_multifc(args.input)
    print("[PASS] Outer/data/reference ZIP CRCs, UTF-8, literal-tab fields and ratings verified")
    print("[PASS] Test reference labels joined in original row order with length validation")
    expected_counts = dict(train=27940, dev=3493, test=3491)
    counts = dict(Counter(r["split"] for r in rows))
    check(counts == expected_counts, f"Audited release split counts: {counts}")
    check(len({r["id"] for r in rows}) == len(rows), "IDs unique across all splits")
    by_text = defaultdict(list)
    for row in rows:
        row.pop("dataset")
        row["empty_text"] = not bool(row["text"])
        row["replacement_characters"] = row["text_raw"].count("\ufffd")
        row["word_count"] = len(row["text"].split())
        row["group_id"] = ("claim-" + hashlib.sha256(key(row["text"]).encode()).hexdigest()
                           if row["text"] else "empty-" + row["id"])
        if row["text"]:
            by_text[key(row["text"])].append(row)
    check(all(clean(r["text"]) == r["text"] for r in rows), "Claim normalization is stable")
    duplicates = [dict(ids=[r["id"] for r in group],
                       splits=sorted({r["split"] for r in group}),
                       labels=sorted({r["label"] for r in group}))
                  for group in by_text.values() if len(group) > 1]
    overlaps = {}
    for left, right in [("train", "dev"), ("train", "test"), ("dev", "test")]:
        shared = [key for key, group in by_text.items()
                  if {left, right}.issubset({r["split"] for r in group})]
        overlaps[f"{left}_{right}"] = len(shared)
        print(f"{'[WARN]' if shared else '[PASS]'} Nonempty duplicate claims {left}/{right}: {len(shared)}")
    damaged = [r["id"] for r in rows if r["replacement_characters"]]
    domains = sorted({r["domain"] for r in rows})
    check(len(domains) == 26, "26 source domains present")
    print("Source domain       train     dev    test")
    for domain in domains:
        c = Counter(r["split"] for r in rows if r["domain"] == domain)
        print(f"  {domain:15} {c['train']:7} {c['dev']:7} {c['test']:7}")
    stats = {}
    for split in ("train", "dev", "test"):
        group = [r for r in rows if r["split"] == split]
        empty = [r["id"] for r in group if r["empty_text"]]
        labels = Counter(r["label"] for r in group)
        nulls = {field: sum(r[field] in ("", "None", "['None']") for r in group) for field in META}
        lengths = [r["word_count"] for r in group]
        print(f"{split}: {len(group):,} records; {len(labels)} label strings; {len(empty)} empty claims")
        if empty:
            print(f"[WARN] Empty claims retained and flagged: {', '.join(empty)}")
        print(f"  Most frequent labels: {labels.most_common(5)}")
        print(f"  Word counts (whitespace): min={min(lengths)}, mean={sum(lengths)/len(lengths):.1f}, max={max(lengths)}")
        print(f"  Missing metadata: {nulls}")
        example = next(r for r in group if r["text"])
        print(f"  Example: {example['id']} | {example['label']} | {example['text'][:150]!r}")
        stats[split] = dict(rows=len(group), label_counts=dict(labels), empty_claim_ids=empty,
                            missing_metadata_counts=nulls,
                            unicode_line_separator_count=sum(r["text_raw"].count("\u2028") for r in group),
                            domain_counts=dict(Counter(r["domain"] for r in group)),
                            domain_label_counts={domain: dict(Counter(r["label"] for r in group if r["domain"] == domain)) for domain in domains})
    print("[WARN] Release has 34,924 claims; original paper reports 34,918.")
    print("[WARN] Reason/title metadata can reveal verdicts. Model input must use the text column only for claim-only work.")
    print("[INFO] Native ratings retained. Additional binary exports use 0=real, 1=fake; uncertain/mixed ratings are excluded.")
    check(before == sha256(args.input), "Original ZIP SHA-256 unchanged")
    report = dict(dataset="MultiFC", input_path=str(args.input.resolve()), input_sha256=before,
                  original_unchanged=True, splits=stats, duplicate_claim_groups=duplicates,
                  cross_split_nonempty_overlap=overlaps, replacement_character_ids=damaged,
                  normalization="Claim only; NFC and whitespace; preserve case, punctuation, numbers and negation",
                  label_policy="Native exports retain ratings; binary exports use source-aware 0=real, 1=fake rules",
                  duplicate_policy="Casefold and whitespace-normalized exact text; no semantic near-duplicate check; official splits unchanged",
                  test_labels="Positional reference join before processing; equal lengths cannot independently prove semantic alignment",
                  evidence="No retrieved evidence corpus in this release")
    def produce(output):
        profiles = {"official": {}, "leakage_controlled": {}, "binary": {}}
        binary_rows_by_split, binary_exclusions = {}, []
        names, exclusions = [], []
        seen_prior_splits = set()
        derived_counts = {}
        train_label_keys = {row["domain"] + "::" + row["label"] for row in rows
                            if row["split"] == "train" and row["text"]}
        unseen_labels = {}
        for split in ("train", "dev", "test"):
            group = [row for row in rows if row["split"] == split]
            csv_name = f"{split}.csv"
            csv_roundtrip(output / csv_name, group, FIELDS)
            names.append(csv_name)
            official = [model_record(row) for row in group if row["text"]]
            derived = []
            for row in group:
                if not row["text"]:
                    exclusions.append(dict(id=row["id"], split=split, reason="empty_claim", profiles="both"))
                elif key(row["text"]) in seen_prior_splits:
                    exclusions.append(dict(id=row["id"], split=split, reason="matches_earlier_split", profiles="leakage_controlled"))
                else:
                    derived.append(model_record(row))
            # Only earlier partitions influence filtering; the test partition never changes train/dev.
            seen_prior_splits.update(key(row["text"]) for row in group if row["text"])
            for profile, records in [("official", official), ("leakage_controlled", derived)]:
                name = f"{profile}_{split}.jsonl"
                write_jsonl(output / name, records)
                names.append(name)
                profiles[profile][split] = name
            derived_counts[split] = len(derived)
            unseen_labels[split] = sorted({r["label_key"] for r in official} - train_label_keys)
            # Binary selection uses the overlap-controlled subset, then removes mixed/uncertain ratings.
            derived_ids = {row["id"] for row in derived}
            eligible = [binary_record(row) for row in group if row["id"] in derived_ids
                        and binary_decision(row["domain"], row["label"])[0] is not None]
            targets_by_text = defaultdict(set)
            for record in eligible:
                targets_by_text[key(record["text"])].add(record["label_id"])
            conflicts = {text for text, targets in targets_by_text.items() if len(targets) > 1}
            binary_rows = []
            for row in group:
                target, reason = binary_decision(row["domain"], row["label"])
                if not row["text"]:
                    reason = "empty_claim"
                elif target is None:
                    pass
                elif row["id"] not in derived_ids:
                    reason = "matches_earlier_split"
                elif key(row["text"]) in conflicts:
                    reason = "conflicting_binary_labels_within_split"
                else:
                    binary_rows.append(binary_record(row))
                    continue
                binary_exclusions.append({"id": row["id"], "split": split, "domain": row["domain"],
                                          "original_label": row["label"], "reason": reason})
            check(all(type(r["label"]) is int and r["label"] == r["label_id"] and r["label"] in (0, 1)
                      for r in binary_rows), f"{split}: binary labels are integers 0=real, 1=fake")
            check({r["label_id"] for r in binary_rows} == {0, 1}, f"{split}: both binary classes present")
            for suffix in ["csv", "jsonl"]:
                name = f"binary_{split}.{suffix}"
                if suffix == "csv":
                    csv_roundtrip(output / name, binary_rows, BINARY_FIELDS)
                else:
                    write_jsonl(output / name, binary_rows)
                    profiles["binary"][split] = name
                names.append(name)
            binary_rows_by_split[split] = binary_rows
            print(f"[BINARY] {split}: {len(binary_rows):,}/{len(group):,} rows retained; "
                  f"real={sum(r['label_id'] == 0 for r in binary_rows):,}, "
                  f"fake={sum(r['label_id'] == 1 for r in binary_rows):,}")
        sets = []
        for split in ("train", "dev", "test"):
            with (output / profiles["leakage_controlled"][split]).open(encoding="utf-8") as handle:
                sets.append({key(json.loads(line)["text"]) for line in handle})
        check(not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]),
              "Derived profile has zero normalized exact claim overlap across splits")
        report["derived_profile_counts"] = derived_counts
        report["model_input_exclusions"] = exclusions
        report["heldout_domain_label_keys_not_in_train"] = unseen_labels
        report["profile_policy"] = ("Full CSVs preserve every original row/order. Official JSONL keeps original membership/order "
                                    "but excludes empty claims. Derived JSONL also filters heldout text seen in earlier splits "
                                    "(train > dev > test); within-split duplicates remain. No semantic near-duplicate check.")
        binary_sets = [{key(r["text"]) for r in binary_rows_by_split[split]}
                       for split in ("train", "dev", "test")]
        check(not (binary_sets[0] & binary_sets[1] or binary_sets[0] & binary_sets[2] or binary_sets[1] & binary_sets[2]),
              "Binary profiles have zero normalized exact claim overlap across splits")
        retained_ids = {r["id"] for group in binary_rows_by_split.values() for r in group}
        check(len(retained_ids) + len(binary_exclusions) == len(rows), "Every input row is retained or has an exclusion reason")
        mapping_rows = []
        pairs = sorted({(row["domain"], row["label"]) for row in rows})
        for domain, label in pairs:
            target, reason = binary_decision(domain, label)
            counts = Counter(row["split"] for row in rows if row["domain"] == domain and row["label"] == label)
            mapping_rows.append(dict(domain=domain, original_label=label,
                                     action="exclude" if target is None else "keep",
                                     label_id="" if target is None else target, reason=reason,
                                     train=counts["train"], dev=counts["dev"], test=counts["test"]))
        csv_roundtrip(output / "binary_mapping.csv", mapping_rows, list(mapping_rows[0]))
        csv_roundtrip(output / "binary_excluded.csv", binary_exclusions,
                      ["id", "split", "domain", "original_label", "reason"])
        names.extend(["binary_mapping.csv", "binary_excluded.csv"])
        binary_domains = {r["domain"] for group in binary_rows_by_split.values() for r in group}
        report["binary"] = dict(policy=BINARY_POLICY, meaning={"0": "real", "1": "fake"},
                                majority_rating_policy="mostly true/correct -> real; mostly false/fiction -> fake",
                                sources=BINARY_SOURCES, original_rows=len(rows),
                                retained_rows=len(retained_ids), retained_percent=round(100 * len(retained_ids) / len(rows), 2),
                                original_domains=len(domains), retained_domains=sorted(binary_domains),
                                excluded_domains=sorted(set(domains) - binary_domains),
                                exclusion_counts=dict(Counter(r["reason"] for r in binary_exclusions)),
                                splits={split: dict(rows=len(group),
                                                    label_counts=dict(Counter(r["label_id"] for r in group)),
                                                    domain_label_counts={domain: dict(Counter(r["label_id"] for r in group if r["domain"] == domain))
                                                                         for domain in sorted(binary_domains)})
                                        for split, group in binary_rows_by_split.items()})
        check(all(binary_decision(r["domain"], r["original_label"])[0] == r["label_id"]
                  for group in binary_rows_by_split.values() for r in group), "Every binary target matches its source-specific rule")
        print(f"[BINARY] Total retained: {len(retained_ids):,}/{len(rows):,}; source coverage: {len(binary_domains)}/26")
        print(f"[BINARY] Excluded source codes: {', '.join(sorted(set(domains) - binary_domains))}")
        single_class = sorted(domain for domain in binary_domains
                              if len({r["label_id"] for r in binary_rows_by_split["train"] if r["domain"] == domain}) == 1)
        report["binary"]["single_class_train_domains"] = single_class
        report["binary"]["domain_caveat"] = "Source codes are fact-checking websites, not topics; retained sources may have only one class or no test rows"
        print(f"[WARN] Single-class training sources: {', '.join(single_class)}. See per-source counts before evaluation.")
        print("[WARN] Binary classes are imbalanced; use macro-F1, per-class/source metrics and a majority baseline.")
        print(f"[INFO] Derived leakage-controlled counts: {derived_counts}")
        if any(unseen_labels.values()):
            print("[WARN] Heldout source/label combinations absent from nonempty training rows: see quality report")
        names.append("quality_report.json")
        write_json(output / "quality_report.json", report)
        check(before == sha256(args.input), "Original ZIP unchanged after saving results")
        return dict(dataset="MultiFC", input_sha256=before, profiles=profiles,
                    files=file_manifest(output, names),
                    next_stage={"status": "binary_ready", "default_profile": "binary", "feature_text": "text",
                                "target": "label_id", "native_rating": "original_label", "source_domain": "domain",
                                "binary_mapping": BINARY_POLICY,
                                "fit_vocabulary_on": "train only", "test": "evaluation only; never select hyperparameters",
                                "reference_repository": "Derived topic-domain/binary dataset is not this original release"})
    publish_outputs(args.output_dir, produce, args.input)
    print("Use binary_train/dev/test.csv or load_preprocessed() for 0=real, 1=fake. Native audit files remain available.")


if __name__ == "__main__":
    run()
