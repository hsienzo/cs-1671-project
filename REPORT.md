# Progress Report: Part-of-Speech Tagging of Pittsburgh Interview Speech

## 1. Data
- **Source:** two APLS interviews, with one word per row in `POS-tagging/*_with_golden_manual.csv`.
- **Gold tags:** hand-checked tags in the Penn Treebank tagset. The gold data uses 33 tags.

| Interview | Speaker | Words | Gold-tagged words |
|---|---|---|---|
| CB01 | white speaker | 11,271 | 3,375 |
| HD20 | Black speaker | 8,855 | 1,013 |

- **Imbalance:** the data is heavily imbalanced. Fillers (UH: "um", "mmm", "like") are the most common tag at about 13%, and the 5 most common tags cover about 50% of words. Several tags appear fewer than 10 times.
- **Contractions:** contractions carry two gold tags, e.g. "i'm" = `PRP VBP`.

## 2. Method

### Baselines (used as-is, no training)
| Tagger | Type |
|---|---|
| Stanford left3words | Maximum-entropy model that reads left to right (2003) |
| Stanford bidirectional | Maximum-entropy model that looks both ways; **the tagger APLS uses** |
| Flair `pos` | Bidirectional LSTM with a CRF layer, using character-level embeddings |
| QCRI BERT | Multilingual BERT already fine-tuned for POS tagging on Penn Treebank newspaper text |

### Main approach: linear probe on frozen BERT / RoBERTa
There are only about 4,400 gold words, too few to fine-tune a 110-million-weight model. So the pretrained model stays **frozen**, and only a small linear layer is trained on top of it.

1. **Input:** each speaker turn is passed whole, for context. Contractions are split with NLTK's Treebank tokenizer, the same convention as the original Stanford run (e.g. "i'm" → "i" + "'m").
2. **Features:** frozen `bert-base-cased` or `roberta-base`. Each piece is represented by the **mean of the last 4 hidden layers** at its **first subword**, giving 768 numbers.
3. **Classifier:** one `torch.nn.Linear(768, 33)` layer (about 25k weights), trained with cross-entropy and AdamW (learning rate 1e-2, weight decay 1e-2, 300 full-batch epochs) on standardized features.
4. **Evaluation:**
   - **5-fold cross-validation grouped by turn:** each gold word is tagged by a layer that never saw its turn.
   - **Cross-interview check:** train on one interview, test on the other.
5. **Scoring:** exact-match accuracy per word, the same rule as the original `get_accuracy_scores.py`. Piece tags are joined back together (e.g. `PRP VBP`) before comparing with the gold tag.

The probe uses about 80% of the gold data for training in each fold, while the baselines use none of it. This is the probe's advantage over the baselines.

## 3. Main Result
**Accuracy (%) on all gold words.**

| Tagger | CB01 | HD20 | Pooled |
|---|---|---|---|
| Stanford left3words | 91.56 | 89.93 | 91.18 |
| Stanford bidirectional (APLS) | 91.88 | 91.31 | 91.75 |
| **Flair** | **96.30** | **94.77** | **95.94** |
| QCRI BERT (as-is) | 87.08 | 84.70 | 86.53 |
| BERT + linear (5-fold CV) | 91.73 | 89.73 | 91.27 |
| RoBERTa + linear (5-fold CV) | 89.33 | 87.66 | 88.95 |

The Stanford and Flair numbers exactly reproduce the original APLS evaluation, which confirms the scoring is consistent.

**Cross-interview (linear probe):**

| Model | Train → Test | Accuracy |
|---|---|---|
| BERT + linear | CB01 → HD20 | 87.86 |
| BERT + linear | HD20 → CB01 | 84.80 |
| RoBERTa + linear | CB01 → HD20 | 86.48 |
| RoBERTa + linear | HD20 → CB01 | 81.45 |

**Summary:**
- Flair is still the best tagger.
- Frozen BERT with a linear layer matches the Stanford tagger APLS uses (91.27 vs 91.75 pooled). RoBERTa is about 2 points behind BERT.
- The ready-made QCRI BERT tagger is the worst model.

## 4. Error Analysis

### Per-tag F1 (%), 14 most common gold tags
| Tag | n | Stanford L3W | Stanford Bi | Flair | QCRI | BERT+lin | RoBERTa+lin |
|---|---|---|---|---|---|---|---|
| UH | 566 | 84.3 | 85.1 | 98.7 | 65.4 | 94.9 | 90.6 |
| PRP | 535 | 99.6 | 99.6 | 99.8 | 98.9 | 99.2 | 97.5 |
| NN | 407 | 88.7 | 81.4 | 90.8 | 75.2 | 85.2 | 83.3 |
| RB | 346 | 94.3 | 93.5 | 94.2 | 83.2 | 85.8 | 85.4 |
| IN | 340 | 88.0 | 87.7 | 93.0 | 82.2 | 89.3 | 87.6 |
| DT | 277 | 95.1 | 97.1 | 98.5 | 97.1 | 93.9 | 91.9 |
| VBD | 272 | 99.1 | 99.4 | 99.8 | 99.4 | 97.6 | 96.0 |
| JJ | 185 | 93.0 | 94.9 | 96.0 | 89.1 | 80.3 | 72.8 |
| CC | 182 | 99.7 | 100.0 | 99.5 | 100.0 | 99.7 | 92.9 |
| VB | 167 | 88.8 | 91.5 | 97.6 | 85.6 | 91.4 | 90.9 |
| VBP | 164 | 91.2 | 96.0 | 97.6 | 82.6 | 94.9 | 90.3 |
| NNS | 147 | 98.6 | 97.2 | 99.3 | 99.3 | 94.1 | 93.0 |
| PRP$ | 106 | 98.6 | 99.0 | 99.5 | 99.0 | 99.0 | 96.7 |
| NNP | 74 | 10.5 | 13.0 | 47.4 | 2.7 | 54.7 | 62.6 |

### Finding 1: Fillers separate the models
| Accuracy (%) | UH words | All other words |
|---|---|---|
| Stanford bidirectional | 74.6 | 94.3 |
| Flair | 99.3 | 95.4 |
| QCRI BERT | **49.3** | 92.0 |
| BERT + linear | **96.3** | 90.5 |

- **QCRI misses half the fillers:** its most-missed words are `mmm` (89 errors), `hmm` (85), `like` (26), `um` (21) and `okay` (16), mostly tagged as NN. QCRI was fine-tuned on newspaper text, which has almost no fillers. On other words it gets 92%, close to Stanford.
- **The linear probe learns fillers from the interviews themselves:** it reaches 96.3% on UH. This is the main reason it beats QCRI, even though both are built on BERT.

### Finding 2: Proper nouns fail because the transcripts are lowercase
- **NNP is the weakest tag for every model** (F1 2.7–62.6). The transcripts are all lowercase, including "i", so capitalization can't signal a name.
- **The probe's top confusions are NN↔NNP** (23 and 22 errors). "hill", from "the Hill District", a Pittsburgh neighbourhood, is among its most-missed words.
- **The probe still beats the baselines on NNP** (BERT 54.7, RoBERTa 62.6 vs Stanford 13.0), probably because it learned the local place names from the training folds.

### Finding 3: The probe is weaker on ambiguous function words and adjectives
- **Most-missed words for BERT + linear:** `to` (17), `so` (11), `that` (11), `all` (9).
  - These words switch roles: "to" can be TO or IN, "so" can be RB, IN or UH, "that" can be DT, IN or WDT, and "all" can be DT or RB.
  - The corresponding confusions are IN↔RB and IN→TO.
- **Adjectives are weak:** JJ F1 is 80.3 for BERT and 72.8 for RoBERTa, against 96.0 for Flair. The errors are mostly JJ↔NN.
- **Common, easy tags are slightly worse too:** on RB, DT and NNS, the probe trails the baselines by 3–8 points. These are classic signs of a classifier trained on too little data.

### Finding 4: BERT and Flair make different mistakes, but Flair makes fewer
On the 4,388 gold words:
- BERT + linear is right where Flair is wrong on **88 words**.
- Flair is right where BERT + linear is wrong on **293 words**.

### Example: CB01, turn 1, first 12 words
| Word | Gold | Flair | QCRI | BERT + linear |
|---|---|---|---|---|
| manipulate | VB | VB | VB | VB |
| someone | NN | NN | NN | NN |
| now | RB | RB | RB | RB |
| um | UH | UH | **IN** | UH |
| i | PRP | PRP | PRP | PRP |
| wanna | VBP TO | **VBP** | VBP TO | VBP TO |
| make | VB | VB | VB | VB |
| sure | JJ | JJ | JJ | **IN** |
| i | PRP | PRP | PRP | PRP |
| have | VBP | VBP | VBP | VBP |
| start | VB | VB | VB | **NNP** |
| with | IN | IN | IN | IN |

Each model makes a different kind of mistake:
- **QCRI** mistakes a filler.
- **Flair** gives "wanna" only one tag. Flair doesn't split contractions, so it can never match a two-tag gold label.
- **The probe** misreads ordinary words.

### Unavoidable errors
Five gold words can't be matched by any model because their number of pieces differs from their number of gold tags: "lotta" (×3), "kinda" and "parent's". This caps the maximum accuracy at 99.91% for CB01 and 99.80% for HD20.

## 6. Challenges and Next Steps
1. **Overfitting:** the linear layer reached a training loss of 0.001 in every fold, so it memorized the training data. That explains its weakness on common tags.
   - **Next step:** stronger weight decay (0.1–1) and input dropout, chosen by cross-validation.
2. **Little gold data:** there are 4,569 training pieces in total, and transfer between interviews is weak. Training on HD20, the smaller interview (1,013 gold words), drops accuracy on CB01 to 84.8% (BERT) and 81.5% (RoBERTa).
   - **Next step:** hand-label more turns, especially from HD20.
3. **Gold-tag bias:** the gold tags were made by correcting machine output, so they may favour the original taggers.
4. **Lowercase input:** this hurts the cased models (QCRI, bert-base-cased), especially on proper nouns.
   - **Next step:** compare against an uncased BERT.

## Reproducing these results
See `README.md`.
- **Main results and cross-interview table:** `POS-tagging/bert_roberta_pos_eval.ipynb`, run on Colab with an NVIDIA A100.
- **Section 5 numbers:** `python POS-tagging/error_analysis.py`.
