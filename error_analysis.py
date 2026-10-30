"""Error analysis for all six taggers, computed from the notebook's saved predictions.

Reads the *_with_transformer_preds.csv files written by bert_roberta_pos_eval.ipynb.
Needs only pandas; run from anywhere:  python POS-tagging/error_analysis.py
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = {"CB01": "CB01interview1_with_transformer_preds.csv",
         "HD20": "HD20interview1_with_transformer_preds.csv"}
MODELS = {"Stanford left3words": "pos_tags_left3words",
          "Stanford bidirectional": "pos_tags_bidirectional",
          "Flair": "flair_pos_tags",
          "QCRI BERT": "qcri_bert",
          "BERT + linear": "bert_linear_cv",
          "RoBERTa + linear": "roberta_linear_cv"}
TRANSFORMERS = ["qcri_bert", "bert_linear_cv", "roberta_linear_cv"]

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

dfs = {n: pd.read_csv(os.path.join(HERE, f)) for n, f in FILES.items()}
gold = {n: d[d["golden"].notna()] for n, d in dfs.items()}
g = pd.concat(gold.values())


def acc(df, col, mask=None):
    df = df if mask is None else df[mask]
    return 100 * (df[col] == df["golden"]).mean()


def f1(tag, col):
    pred, true = g[col] == tag, g["golden"] == tag
    tp = (pred & true).sum()
    if tp == 0:
        return 0.0
    p, r = tp / pred.sum(), tp / true.sum()
    return 100 * 2 * p * r / (p + r)


print("== Accuracy (%), exact match on gold words")
table = pd.DataFrame({**{n: [acc(d, c) for c in MODELS.values()] for n, d in gold.items()},
                      "Pooled": [acc(g, c) for c in MODELS.values()]}, index=list(MODELS))
print(table.round(2), "\n")

print("== Accuracy (%) on fillers (gold UH) vs all other words")
uh = g["golden"] == "UH"
print(pd.DataFrame({"UH": [acc(g, c, uh) for c in MODELS.values()],
                    "non-UH": [acc(g, c, ~uh) for c in MODELS.values()]},
                   index=list(MODELS)).round(1), "\n")

print("== Per-tag F1 (%), 14 most frequent gold tags")
top = g["golden"].value_counts().head(14)
print(pd.DataFrame({"n": top, **{m: [f1(t, c) for t in top.index] for m, c in MODELS.items()}}).round(1), "\n")

for col in TRANSFORMERS:
    wrong = g[g[col] != g["golden"]]
    print(f"== {col}: top 10 confusions (gold -> predicted)")
    print(wrong.groupby(["golden", col]).size().nlargest(10).to_string(), "\n")
    print(f"== {col}: words with the most errors")
    print(wrong["token"].value_counts().head(12).to_string(), "\n")

print("== QCRI BERT: filler words (gold UH) it mis-tagged")
print(g[uh & (g["qcri_bert"] != "UH")]["token"].value_counts().head(8).to_string(), "\n")

b_ok, f_ok = g["bert_linear_cv"] == g["golden"], g["flair_pos_tags"] == g["golden"]
print("== BERT + linear vs Flair")
print(f"BERT right, Flair wrong: {(b_ok & ~f_ok).sum()}   Flair right, BERT wrong: {(f_ok & ~b_ok).sum()}\n")

print("== Example: first 12 words of CB01 turn 1")
print(dfs["CB01"].head(12)[["token", "golden", "flair_pos_tags", "qcri_bert", "bert_linear_cv"]].to_string(index=False))
