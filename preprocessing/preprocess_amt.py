"""Preprocess FakeNewsAMT articles only; Celebrity is excluded. Python 3.10+; no external packages."""

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
DATASET = 'amt'

TOPICS = dict(biz="business", edu="education", entmt="entertainment",
              polit="politics", sports="sports", tech="technology")
FIELDS = ["id", "label", "label_id", "domain", "group_id", "split", "text",
          "text_raw", "title", "body", "source_file", "replacement_characters", "word_count"]

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


def load_amt(path):
    rows = []
    with checked_zip(path) as archive:
        for name in sorted(archive.namelist()):
            if not re.fullmatch(r"fakeNewsDatasets/fakeNewsDataset/(fake|legit)/[^/]+\.txt", name):
                continue  # Celebrity is a separate dataset, deliberately excluded.
            label = name.split("/")[-2]
            match = re.fullmatch(r"(biz|edu|entmt|polit|sports|tech)(\d+)\." + label + r"\.txt", Path(name).name)
            if not match:
                raise ValueError(f"Unexpected AMT filename: {name}")
            topic, number = match.groups()
            group = f"{topic}-{int(number):03d}"
            raw = archive.read(name).decode("utf-8")
            rows.append(dict(dataset="FakeNewsAMT", id=f"{group}-{label}",
                             split="unsplit", label=label, domain=TOPICS[topic],
                             group_id=group, text_raw=raw, text=clean(raw), source_file=name))
    counts = Counter((r["domain"], r["label"]) for r in rows)
    if len(rows) != 480 or any(counts[(d, label)] != 40
                               for d in TOPICS.values() for label in ("fake", "legit")):
        raise ValueError(f"Unexpected AMT distribution: {counts}")
    groups = defaultdict(set)
    for row in rows:
        groups[row["group_id"]].add(row["label"])
    if len(groups) != 240 or any(labels != {"fake", "legit"} for labels in groups.values()):
        raise ValueError("AMT filename pair identifiers are incomplete")
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
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "amt")
    parser.add_argument("--make-splits", action="store_true", help="Also export derived group-stratified 80/10/10 AMT splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.input = args.input if args.input is not None else find_input('fakeNewsAMTDatasets.zip')
    print("FakeNewsAMT: article preprocessing")
    print(f"Input: {args.input.resolve()}")
    check(args.input.is_file(), "Input ZIP exists")
    before = sha256(args.input)
    rows = load_amt(args.input)
    print("[PASS] ZIP CRC, UTF-8 decoding, filenames, 480 records and 240 filename pairs verified")
    for row in rows:
        row.pop("dataset")
        row["label_id"] = {"legit": 0, "fake": 1}[row["label"]]
        row["replacement_characters"] = row["text_raw"].count("\ufffd")
        row["word_count"] = len(row["text"].split())
        title, _, body = row["text_raw"].replace("\r\n", "\n").partition("\n")
        row["title"], row["body"] = clean(title), clean(body)
    check(len({r["id"] for r in rows}) == len(rows), "Article IDs are unique")
    check(all(r["text"] for r in rows), "Every article contains text")
    check(all(clean(r["text"]) == r["text"] for r in rows), "Text normalization is stable")
    labels = dict(Counter(r["label"] for r in rows))
    print(f"Labels: {labels}; numeric mapping: legit=0, fake=1")
    print("Topic counts (fake / legit):")
    for topic in sorted(TOPICS.values()):
        counts = Counter(r["label"] for r in rows if r["domain"] == topic)
        print(f"  {topic:15} {counts['fake']:3} / {counts['legit']:3}")
    by_text = defaultdict(list)
    for row in rows:
        by_text[key(row["text"])].append(row["id"])
    duplicates = [ids for ids in by_text.values() if len(ids) > 1]
    damaged = [r["id"] for r in rows if r["replacement_characters"]]
    if damaged:
        print(f"[WARN] Existing replacement characters in: {', '.join(damaged)}")
    print(f"{'[WARN]' if duplicates else '[PASS]'} Normalized duplicate groups: {len(duplicates)}")
    lengths = [r["word_count"] for r in rows]
    print(f"Word counts (whitespace tokens): min={min(lengths)}, mean={sum(lengths)/len(lengths):.1f}, max={max(lengths)}")
    print("Example (headline and body retained):")
    print(f"  {rows[0]['id']} | {rows[0]['label']} | {rows[0]['text'][:180]!r}")
    check(before == sha256(args.input), "Original ZIP SHA-256 unchanged")
    report = dict(dataset="FakeNewsAMT", rows=len(rows), labels=labels,
                  domains=dict(Counter(r["domain"] for r in rows)),
                  input_path=str(args.input.resolve()), input_sha256=before,
                  original_unchanged=True, replacement_character_ids=damaged,
                  normalized_duplicate_groups=duplicates,
                  word_counts=dict(min=min(lengths), mean=sum(lengths)/len(lengths), max=max(lengths)),
                  normalization="NFC and whitespace only; retain headline, body, case, punctuation, numbers and negation",
                  split="unsplit", label_mapping={"legit": 0, "fake": 1},
                  group_policy="Filename topic/number identifies candidate pairs, not manually verified semantic pairs; keep groups together in any later split.",
                  celebrity="Excluded; separate collection")
    def produce(output):
        csv_roundtrip(output / "articles.csv", rows, FIELDS)
        write_jsonl(output / "model_input.jsonl", [model_record(row) for row in rows])
        profiles = {"official": {"unsplit": "model_input.jsonl"}}
        names = ["articles.csv", "model_input.jsonl", "quality_report.json"]
        if args.make_splits:
            assignments = {}
            rng = random.Random(args.seed)
            for domain in sorted(TOPICS.values()):
                groups = sorted({r["group_id"] for r in rows if r["domain"] == domain})
                rng.shuffle(groups)
                for index, group in enumerate(groups):
                    assignments[group] = "train" if index < 32 else "dev" if index < 36 else "test"
            profiles["grouped"] = {}
            for split in ("train", "dev", "test"):
                selected = [dict(model_record(row), split=split) for row in rows
                            if assignments[row["group_id"]] == split]
                name = f"grouped_{split}.jsonl"
                write_jsonl(output / name, selected)
                profiles["grouped"][split] = name
                names.append(name)
            check(len(assignments) == 240, "240 AMT groups assigned exactly once")
            report["derived_split"] = dict(seed=args.seed, ratios=[0.8, 0.1, 0.1],
                                          counts=dict(train=384, dev=48, test=48),
                                          official=False, method="32/4/4 filename pairs per topic")
        write_json(output / "quality_report.json", report)
        check(before == sha256(args.input), "Original ZIP unchanged after saving results")
        return dict(dataset="FakeNewsAMT", input_sha256=before, profiles=profiles,
                    files=file_manifest(output, names),
                    next_stage={"status": "grouped_profile_ready" if args.make_splits else "requires_group_split",
                                "feature_text": "text", "target": "label_id", "group": "group_id",
                                "topic_domain": "domain", "fit_vocabulary_on": "training partition only",
                                "transformer": "Use the chosen model tokenizer; no stemming/stopword deletion",
                                "baseline": "TF-IDF unigram/bigram candidate; fit on training only"})
    publish_outputs(args.output_dir, produce)
    print("No training performed. AMT split exports are optional; review warnings before modeling.")


if __name__ == "__main__":
    run()
