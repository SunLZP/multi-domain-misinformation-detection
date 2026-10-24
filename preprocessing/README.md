# EE6405 — preprocessing and the next-stage handoff

Only two preprocessing scripts. Python 3.10+ and no pip packages. Keep your existing layout:

```text
EE6405 Project/
  fakeNewsAMTDatasets.zip
  MultiFC_Codalab.zip
  preprocessing/
    preprocess_amt.py
    preprocess_multifc.py
    run_preprocessing.cmd
    README.md
    PREPROCESSING_NOTES.md
```

## Run

In VS Code Terminal, go into the folder containing the code:

```powershell
cd "C:\Users\yiwen\OneDrive\Documents\NTU\EE6405 Project\preprocessing"
.\run_preprocessing.cmd amt
.\run_preprocessing.cmd multifc
```

The scripts automatically find the ZIPs one folder above. They also support ZIPs beside the scripts, or in data/ in either location. Explicit --input overrides are optional. No copying or renaming is required. Existing extracted dataset folders are not used or modified.

Double-click run_preprocessing.cmd to run both. It looks for a compatible available Codex Python runtime, then py or python (3.10+ required). It does not install anything or change PATH/security settings. If no runtime is available it explains what is needed. On another machine, Python 3.10+ on PATH is sufficient. With a working interpreter selected, VS Code's Run Python File button also works without path arguments.

## Automatic saving and visible checks

The default is output/amt and output/multifc beside the scripts. If a directory cannot be created or written, the code automatically retries under:

```text
C:\Users\<you>\EE6405-results\<project-name>-<project-id>\amt
C:\Users\<you>\EE6405-results\<project-name>-<project-id>\multifc
```

The project ID is a short hash of the script folder, preventing collisions between different projects. Only results are saved there; your source files and datasets stay where they are. The final [DONE] prints the exact path. If both locations fail, the error lists them. Moving the project changes its ID; rerun preprocessing in the new folder.

[PASS] checks archive CRCs, row structure, IDs, labels, original hashes, full CSV/JSONL readback and manifest hashes. [WARN] identifies known dataset issues. [FAIL] stops the run with a nonzero exit code. [DONE] means output verification completed. A success manifest is invalidated before replacing output files and published only after saving succeeds. Old unused derived files may remain after a different option is run; only files listed in the current manifest belong to that run. Do not use files from a failed run.

## Outputs

| Dataset | Audit / reference export | Next-stage export |
|---|---|---|
| AMT | articles.csv: all 480 articles, raw/normalized text, separate title/body, labels, topic and pair ID | model_input.jsonl: 480 unsplit examples |
| MultiFC | train.csv (27,940), dev.csv (3,493), test.csv (3,491): every original row, original order and metadata | official_train/dev/test.jsonl: 27,939 / 3,492 / 3,490 nonempty claims |
| MultiFC derived evaluation | Same audit CSVs remain intact | leakage_controlled_train/dev/test.jsonl: 27,939 / 3,476 / 3,466; empty claims and heldout matches to earlier splits explicitly excluded |

Each dataset also has quality_report.json and manifest.json. The report lists exclusions, distributions, overlaps and warnings. The manifest records input/script hashes, output file hashes and the exact profile paths. MultiFC JSONL contains claim text only, without reason/title metadata. The profile named official refers to official split membership after excluding empty claims; it is not a byte-identical benchmark export. The full CSVs retain all records.

AMT text includes headline/body, with legit=0 and fake=1. Celebrity is excluded. MultiFC native ratings stay strings; label_key combines source and rating (e.g. pomt::half-true). There is no automatic binary conversion. Lowercase none is a native rating. Missing metadata None and ['None'] remain in CSVs.

## Stage 2 — define splits and the modeling task

This project still performs preprocessing only; training/evaluation scripts are not included or implied to exist.

AMT has no official split. If you want a deterministic starting point, run:

```powershell
.\run_preprocessing.cmd amt-splits
```

This also produces grouped_train/dev/test.jsonl with 384/48/48 articles, grouped by candidate filename pair and balanced per topic/class using seed 42. Use grouped cross-validation later if the research design requires it; this optional holdout split does not reproduce published cross-validation results. Keep all preprocessing/feature tuning inside the chosen training folds.

For MultiFC, choose and report either official membership or the derived overlap-controlled profile. Choose source-aware classification using label_key/domain, or define and review a domain-specific rating mapping. A binary cross-dataset experiment is not ready until that mapping and the different meanings of domain are resolved. Never interpret all non-true ratings as fake unless explicitly reproducing and labeling that experimental choice.

## Stage 3 — connect a training script to these results

Use the importable loader below. It checks every output hash and locates the newest valid run in the project or automatic results folder. It does not require hardcoded output paths. A custom --output-dir can be supplied explicitly as output_dir to the loader.

```python
import preprocess_amt
import preprocess_multifc

# After running amt-splits; raises a clear error if this profile was not created.
amt = preprocess_amt.load_preprocessed(profile="grouped")
amt_train_text = [row["text"] for row in amt["train"]]
amt_train_target = [row["label_id"] for row in amt["train"]]

multifc = preprocess_multifc.load_preprocessed(profile="leakage_controlled")
multifc_train_text = [row["text"] for row in multifc["train"]]
multifc_train_target = [row["label_key"] for row in multifc["train"]]
# domain is source/splitting information; do not append labels or group IDs to text.
```

For a TF-IDF candidate: fit vocabulary, IDF and any feature selection on train only; use dev for choices; evaluate test once after choosing. For a transformer: use its own tokenizer, measure token lengths on train, and choose truncation/padding on train/dev. These scripts' word counts are only whitespace-based diagnostics. No global stopword removal, stemming or punctuation stripping is imposed. Training-only label encoders should handle heldout unseen labels explicitly; per-source label coverage is in the report.

## Stage 4 — evaluation and domain adaptation

Report macro-F1, per-class/per-source results and the majority-class baseline; keep raw counts and exclusions with the metrics. Compare official vs derived profiles explicitly rather than mixing their results. AMT pairs and MultiFC duplicate IDs/groups must not leak across any new folds. Do not compare a claim-only result to an evidence-based result as if the inputs were equal.

Your reference repository uses a derived MultiFC release with five topics, binary ratings and evidence. The downloaded CodaLab bundle uses source domains and original ratings and lacks the retrieved evidence corpus. Reproducing the reference requires its published data/domain assignments; no additional download or substitution happens in this code.

See [PREPROCESSING_NOTES.md](PREPROCESSING_NOTES.md) for the research comparison, rationale and source links.
