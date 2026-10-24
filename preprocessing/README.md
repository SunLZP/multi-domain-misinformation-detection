# EE6405 — preprocessing

Two standalone scripts; Python 3.10+ and no pip packages. The source ZIPs are opened read-only, never extracted or edited. SHA-256 checks verify they remain unchanged.

## Layout and running

```text
multi-domain-misinformation-detection/
  fakeNewsAMTDatasets.zip
  MultiFC_Codalab.zip
  preprocessing/
    preprocess_amt.py
    preprocess_multifc.py
    run_preprocessing.cmd
    README.md
    PREPROCESSING_NOTES.md
    output/                  # generated
```

Open the repository folder in VS Code. In its PowerShell terminal:

```powershell
cd .\preprocessing
.\run_preprocessing.cmd amt-splits
.\run_preprocessing.cmd multifc
```

If using the original project folder instead, first run:

```powershell
cd "C:\Users\yiwen\OneDrive\Documents\NTU\EE6405 Project\preprocessing"
```

The ZIPs are found one folder above automatically. They can also be beside the scripts or in data/ beside/above them. No copying, renaming or unzipping is needed. Existing extracted dataset folders are unused. `amt` exports AMT unsplit; `amt-splits` also creates train/dev/test. `both` runs unsplit AMT and MultiFC. Double-clicking the runner does the same as `both`.

The runner checks the available Codex Python runtime, then py/python. On teammates' machines, install Python 3.10+ and make it available on PATH. It does not install packages or change system settings. On macOS/Linux use `python3 preprocess_amt.py --make-splits` and `python3 preprocess_multifc.py` inside preprocessing/.

## Use these outputs for binary training

Both datasets' **binary** files use the same columns:

`id, text, label, label_id, original_label, domain, group_id, split`

**label and label_id both mean 0=real, 1=fake.** JSONL stores these as integers; when reading CSV, parse the target column as an integer. original_label preserves the publisher's original string. Use **text** as input and **label_id** as the target. Never add original_label, label, label_id or group_id to the input text. Domain is retained for grouping/reporting; source identity can be a shortcut, so a claim-text baseline should not use it as a predictor.

| Dataset | Binary files in output/ | Rows |
|---|---|---|
| AMT unsplit | amt/binary.csv and binary.jsonl | 480: 240 real, 240 fake |
| AMT derived split (`amt-splits`) | amt/binary_train/dev/test.csv and .jsonl | 384 / 48 / 48 |
| MultiFC derived split | multifc/binary_train/dev/test.csv and .jsonl | 19,610 / 2,445 / 2,434 |

AMT retains headline/body, all six topic domains and candidate filename-pair groups. Celebrity is a separate collection and is excluded. AMT has no official train/test split: the optional split uses seed 42 and 32/4/4 pairs per topic. Keep pairs together in any later folds. This convenience split does not reproduce a published evaluation.

MultiFC binary labels use **explicit source-specific rules**, not a generic true/false substring search. Clear true/accurate equivalents and mostly-true/correct ratings map to 0; clear false/fiction equivalents and mostly-false/fiction ratings map to 1. Half-true, mixture, unknown, time-dependent, promise/position and unreviewed ratings are excluded. GossipCop's original score 0 maps to fake=1; its original 10 maps to real=0; intermediate scores are excluded. These are documented project choices for a binary task, not new ground-truth fact checks.

The binary MultiFC subset keeps **24,489/34,924 rows (70.12%) and 23/26 source domains**. abbc, huca and pose have no retained rows under the reviewed rules. Retaining a source does not guarantee both classes or a test example: see the per-source counts. Train is 6,342 real / 13,268 fake; dev 793 / 1,652; test 791 / 1,643. Half-truths have not been silently assigned to fake.

Original split membership/order is preserved among retained rows. Empty claims are excluded. Heldout claims matching normalized text in earlier original partitions are excluded (train > dev > test), with no reassignment or test-driven change to training. Conflicting binary targets for the same normalized text within a split would also be excluded. Other within-split duplicates remain; semantic near duplicates are not detected. This is a derived evaluation subset, not the full original benchmark.

## Audit files and checks

Native-label audit exports remain available: AMT articles.csv/model_input.jsonl; MultiFC train/dev/test.csv and official_*.jsonl or leakage_controlled_*.jsonl. **Their label strings are original ratings, not the numeric binary targets.** Full audit CSVs preserve every original row, raw text and metadata. MultiFC reason/title metadata can reveal verdicts and must not enter a claim-only model.

MultiFC binary_mapping.csv lists all 160 observed source/rating pairs, their keep/exclude decisions and original split counts. binary_excluded.csv lists all 10,435 excluded row IDs and reasons. No excluded row is deleted from the ZIP or full native CSVs.

Each output folder has quality_report.json with counts, warnings and policies, plus manifest.json with source/script/output hashes and profile paths. [PASS] verifies CRCs, schema, IDs, mapping, CSV/JSONL readback, split overlap and hashes. [WARN] reports dataset issues; [FAIL] stops with a nonzero exit code. Only [DONE] confirms a verified completed run. Two AMT articles contain pre-existing replacement characters; the scripts flag them rather than inventing repairs.

Output defaults to preprocessing/output/<dataset>. On an output I/O failure, an automatic fallback is attempted at `C:\Users\<you>\EE6405-results\<script-folder>-<project-id>\<dataset>`. [DONE] prints the actual destination. The loader below finds the newest verified default/fallback run. The input archive may not be inside an output folder. Existing unused split files may remain after running a different option; only files in the current manifest belong to that run.

## Handoff to training

Run preprocessing first. In a training script inside preprocessing/:

```python
import preprocess_amt
import preprocess_multifc

# Requires amt-splits. Default AMT loader returns the unsplit binary dataset.
amt = preprocess_amt.load_preprocessed(profile="binary_grouped")
multifc = preprocess_multifc.load_preprocessed()  # defaults to binary

amt_train_text = [row["text"] for row in amt["train"]]
amt_train_target = [row["label_id"] for row in amt["train"]]
multifc_train_text = [row["text"] for row in multifc["train"]]
multifc_train_target = [row["label_id"] for row in multifc["train"]]
```

The loader checks output hashes and refuses unavailable profiles. A custom --output-dir must be passed explicitly as output_dir when loading. Old outputs require a rerun to create the binary profile. Native profiles remain accessible explicitly as official/leakage_controlled (MultiFC) or official/grouped (AMT).

Training is the next stage; no model is trained here. Fit TF-IDF vocabulary/IDF and feature selection on training only. For transformers use the model's tokenizer; whitespace word counts are diagnostics, not token counts. Preserve negation, case, punctuation and numbers; choose further transformations using train/dev. Use dev for model choices and test once for final evaluation. Report macro-F1, per-class/source results and a majority baseline because MultiFC is imbalanced. Do not oversample dev/test or claim the binary mapping guarantees better accuracy.

**Domain meanings differ:** AMT domain is topic; MultiFC domain is fact-checking source. Numeric labels now match, but that alone does not make the datasets equivalent. The reference repository uses derived topics and evidence unavailable in this CodaLab bundle. Do not describe this source-domain, claim-only experiment as a reproduction of that project.

See [PREPROCESSING_NOTES.md](PREPROCESSING_NOTES.md) for research comparisons and mapping evidence.
