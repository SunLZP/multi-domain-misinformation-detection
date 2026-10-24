"""Preprocess MultiFC claims using its official splits and original ratings. Python 3.10+; no external packages."""

import argparse
import random
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


def clean(text):
    # Preserve case, punctuation, numbers, stopwords and negation.
    return " ".join(unicodedata.normalize("NFC", text).split())


def key(text):
    return clean(text).casefold()


def checked_zip(source):
    archive = zipfile.ZipFile(source)
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
    if manifest.get("schema_version") != 2 or not manifest.get("complete"):
        raise ValueError(f"Incomplete or incompatible manifest: {path}")
    for name, expected in manifest["files"].items():
        target = (path.parent / name).resolve()
        if not target.is_relative_to(path.parent.resolve()):
            raise ValueError("Manifest contains an invalid file path")
        if sha256(target) != expected["sha256"]:
            raise ValueError(f"Generated file changed or incomplete: {target}")
    return manifest


def load_preprocessed(profile="official", output_dir=None):
    """Next-stage API: return {split: records} after verifying completed output hashes.

    Example: import preprocess_amt; data = preprocess_amt.load_preprocessed()
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
        raise ValueError(f"Unavailable profile {profile!r}; choose {list(manifest['profiles'])}")
    result = {}
    for split, relative in manifest["profiles"][profile].items():
        with (directory / relative).open(encoding="utf-8") as handle:
            result[split] = [json.loads(line) for line in handle]
    return result


def publish_outputs(requested, producer):
    """Probe actual writes, recover once outside the project, commit manifest last."""
    failures = []
    destinations = [requested.resolve(), fallback_output(DATASET).resolve()]
    for output in dict.fromkeys(destinations):
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
            manifest.update(schema_version=2, complete=True,
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
    # Only these fields enter the handoff; no fact-check verdict explanations/titles.
    result = {field: row[field] for field in ["id", "text", "label", "domain", "group_id", "split"]}
    if DATASET == "amt":
        result["label_id"] = row["label_id"]
    else:
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
    print("[INFO] Original ratings and missing metadata sentinels preserved; no binary rating mapping.")
    check(before == sha256(args.input), "Original ZIP SHA-256 unchanged")
    report = dict(dataset="MultiFC", input_path=str(args.input.resolve()), input_sha256=before,
                  original_unchanged=True, splits=stats, duplicate_claim_groups=duplicates,
                  cross_split_nonempty_overlap=overlaps, replacement_character_ids=damaged,
                  normalization="Claim only; NFC and whitespace; preserve case, punctuation, numbers and negation",
                  label_policy="Source-specific strings, including lowercase none; no binary mapping",
                  duplicate_policy="Casefold and whitespace-normalized exact text; no semantic near-duplicate check; official splits unchanged",
                  test_labels="Positional reference join before processing; equal lengths cannot independently prove semantic alignment",
                  evidence="No retrieved evidence corpus in this release")
    def produce(output):
        profiles = {"official": {}, "leakage_controlled": {}}
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
        print(f"[INFO] Derived leakage-controlled counts: {derived_counts}")
        if any(unseen_labels.values()):
            print("[WARN] Heldout source/label combinations absent from nonempty training rows: see quality report")
        names.append("quality_report.json")
        write_json(output / "quality_report.json", report)
        check(before == sha256(args.input), "Original ZIP unchanged after saving results")
        return dict(dataset="MultiFC", input_sha256=before, profiles=profiles,
                    files=file_manifest(output, names),
                    next_stage={"status": "requires_source_label_model_policy", "feature_text": "text",
                                "target": "label_key", "native_rating": "label", "source_domain": "domain",
                                "binary_mapping": "Not applied. Source-specific mapping must be reviewed separately.",
                                "fit_vocabulary_on": "train only", "test": "evaluation only; never select hyperparameters",
                                "reference_repository": "Derived topic-domain/binary dataset is not this original release"})
    publish_outputs(args.output_dir, produce)
    print("Full CSVs preserve all records. Model JSONL excludes empty claims and verdict-bearing metadata.")


if __name__ == "__main__":
    run()
