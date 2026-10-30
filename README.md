# CS1671 Project: POS Tagging of APLS Interview Speech

This project compares part-of-speech taggers on hand-checked tags from two sociolinguistic interviews in the [Archive of Pittsburgh Language and Speech (APLS)](https://apls.pitt.edu).
- **Taggers compared:** Stanford (two models), Flair, a ready-made BERT POS tagger (QCRI), and frozen `bert-base-cased` / `roberta-base` with a trained linear layer.
- **Results, method and error analysis:** see **[REPORT.md](REPORT.md)**.



## How to run
1. **Notebook (main results).** This needs a GPU.
   1. Open `POS-tagging/bert_roberta_pos_eval.ipynb` in Google Colab and choose an A100 or L4 runtime.
   2. Put the two `*_with_golden_manual.csv` files in a Google Drive folder and set `DATA_DIR` in the third code cell.
   3. Run all cells. This takes a few minutes. The notebook prints the accuracy and cross-interview tables and saves `*_with_transformer_preds.csv` to `DATA_DIR`.
   - **Requirements:** PyTorch, `transformers`, pandas and nltk. Colab already has all of these except `transformers`, which the first cell installs.
2. **Error analysis.** This runs locally with no GPU and needs only pandas.
   ```
   python POS-tagging/error_analysis.py
   ```
   It reads the two `*_with_transformer_preds.csv` files and prints every table in Section 5 of the report.

## Credits
- The original pipeline scripts and gold data are from the APLS project (summer 2024, Haiwei Hu); see `POS-tagging/README.md`.
- The original scripts need the Stanford POS Tagger and Flair to rerun, but their outputs are already stored in the gold CSVs.
