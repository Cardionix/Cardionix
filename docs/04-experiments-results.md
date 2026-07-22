# Appendix 4 — Experiments and results

> Companion to [§10 of the README](../README.md#10-experiments-and-results). This appendix is the full per-run log behind the summary in the main report: the runs in the order they were executed, the raw Weights & Biases numbers, and the reasoning that connects one run to the next. Where the README states a conclusion, this document shows the measurements that forced it. Every number here is read from a real W&B run or from a saved checkpoint filename; no metric is estimated or invented. Runs are identified by their W&B run id (the eight-character slug), so they can be located directly in the "Cardio Sonix" project.

## 4.0 How to read this appendix

The narrative has one spine: a taxonomy failure, its correction, and the plateau that followed the correction. Three experimental phases map onto it.

1. **Phase A — the five-class taxonomy** (runs `uutty3vj`, `qjaxj05j`). The full PASCAL label space. Both runs fail, and they fail in the same place, which is what makes the failure diagnostic rather than random.
2. **Phase B — the three-class collapse** (run `uf7xi6nb`, then the best run `sfqx9n2b`). Merging the rare classes into a single *abnormal* class. This is the single largest jump in the project.
3. **Phase C — the plateau** (post-`sfqx9n2b`). Architecture and hyperparameter changes stop paying off. The diagnosis that follows, that the binding constraint is data and representation rather than model capacity, is the pivot that reorganised the rest of the work.

Two conventions used throughout:

- **macro-F1** is the primary selection metric. For a problem with $C$ classes, with per-class precision $P_c$ and recall $R_c$,
  $$
  F_1^{(c)} = \frac{2\,P_c R_c}{P_c + R_c},
  \qquad
  \text{macro-}F_1 = \frac{1}{C}\sum_{c=1}^{C} F_1^{(c)} .
  $$
  The macro average weights every class equally regardless of its support. That is deliberate: under the corpus's heavy imbalance a metric that weighted by support (accuracy, weighted-F1) would let the majority *normal* class mask total failure on a rare class. Macro-F1 does not forgive a collapsed minority class, which is exactly the property that makes the Phase A failures visible.
- **"val > train" is not a typo.** Several runs report a validation score above the training score. This is the signature of a *small* validation fold (high-variance estimate on few samples), not of a model that generalises better than it fits. It is called out where it occurs and it is a caution, not a result.

---

## 4.1 Phase A — the five-class taxonomy collapsed

The first serious runs used the full PASCAL five-class taxonomy: *normal*, *murmur*, *extrahls* (extra heart sound), *extrastole* (extrasystole), *artifact*. The class supports in the DHDataset era were badly skewed:

| Class | Support (DHD) |
|---|---:|
| normal | 351 |
| murmur | 129 |
| extrastole | 46 |
| extrahls | 19 |
| artifact | 40 |

Two classes, *extrastole* (n = 46) and *extrahls* (n = 19), are an order of magnitude rarer than *normal*. Under a 0.7 / 0.3 split, the *extrahls* validation fold holds on the order of half a dozen samples. No selection metric can be trusted at that support, and no gradient signal can train a class that thin. Both facts show up in the runs.

### 4.1.1 Run `uutty3vj`

| Metric | Train | Validation |
|---|---:|---:|
| macro-F1 | 0.741 | **0.593** |
| accuracy | — | 0.744 |
| loss | — | 0.711 |

Per-class validation F1:

| Class | Val F1 | Note |
|---|---:|---|
| normal | 0.813 | majority class, learns cleanly |
| murmur | 1.000 | **spurious**: tiny validation fold, a handful of samples all correct |
| artifact | 0.642 | mediocre, the class is heterogeneous and scarce |
| extrahls | 0.286 | collapsing |
| extrastole | 0.222 | collapsing |

The macro-F1 of 0.593 is an average of one solid class, one spurious perfect class, one mediocre class, and two near-dead classes. The `murmur = 1.000` entry is the trap: read naively it says the model has *solved* murmur detection, but it reflects a validation fold too small to be an estimate of anything. The train macro-F1 (0.741) sitting well above the val macro-F1 (0.593) is the ordinary overfitting direction here, consistent with a model that has memorised the majority classes and has nothing to learn the minority ones from.

### 4.1.2 Run `qjaxj05j`

| Metric | Train | Validation |
|---|---:|---:|
| macro-F1 | — | **0.512** |
| accuracy | — | 0.701 |
| loss | — | 0.820 |

Per-class validation F1:

| Class | Val F1 | Note |
|---|---:|---|
| normal | 0.786 | still the only reliably learned class |
| murmur | 0.968 | high, but again a small-fold artefact, not stability |
| artifact | 0.542 | worse than `uutty3vj` |
| extrahls | 0.267 | collapsing |
| extrastole | **0.000** | **total collapse**: the class is never correctly predicted |

`qjaxj05j` is the same failure as `uutty3vj`, one notch worse. The `extrastole` F1 has gone to **exactly zero**: across the whole validation fold, not a single extrasystole recording is both predicted and correct. A zero F1 is not noise around a low mean; it is a class that the classifier has effectively dropped from its output distribution. The higher validation loss (0.820 vs 0.711) and lower macro-F1 (0.512 vs 0.593) confirm the run is on the worse side of the same instability.

### 4.1.3 What the two runs say together

The two runs are not two data points on a search; they are one diagnosis stated twice. Across both:

- The two rarest classes (*extrastole*, *extrahls*) are always the two lowest F1 scores, and their scores range from bad (0.29) to nonexistent (0.00).
- The two apparently strong minority results (`murmur = 1.000`, `murmur = 0.968`) are both explained by fold size, not by learning, and they move by 0.03 between runs on the same class for no modelled reason, which is exactly the volatility a tiny fold produces.
- The *macro* average is dragged from a respectable-looking accuracy (0.70–0.74) down to 0.51–0.59 entirely by the collapsed minority classes. This is macro-F1 doing its job: refusing to report success while two of five classes are dead.

The mechanism is data starvation, not model deficiency. A network with more capacity cannot manufacture gradient signal for a class with nineteen total examples. The correct move is therefore not a bigger network; it is to change the label space so that scarce, clinically adjacent labels stop competing for data they cannot each support.

<div align="center">
<img src="assets/fig_taxonomy_pivot.png" width="880" alt="Five-class collapse versus three-class recovery, with per-class F1 of the five-class run."/>
</div>

---

## 4.2 Phase B — collapsing the taxonomy

### 4.2.1 The reframing

For a screening instrument, the five-class distinction is not the one that matters. A murmur, an extrasystole, and an extra heart sound are, for triage, the same message: *this heart does not sound normal, refer it*. The class-management layer of the framework (a configuration-level merge rule, no code change) was set to

```
abnormal = { murmur, extrahls, extrastole }
healthy  = { normal }
artifact = { artifact }
```

This collapses the two starved classes into a single *abnormal* class that now aggregates their supports (129 + 46 + 19 = 194 in the DHD era, and far more once PhysioNet is folded in). The scarce-data problem is not solved by adding data here; it is solved by no longer asking the model to *separate* labels it does not have the data to separate.

### 4.2.2 Run `uf7xi6nb` — the intermediate three-class run (pre-PhysioNet)

Before the PhysioNet expansion, the three-class collapse was tested on the smaller corpus. The label space here is *abnormal / artifact / normal*.

| Metric | Validation |
|---|---:|
| macro-F1 | **0.733** |
| accuracy | 0.658 |

Per-class validation F1:

| Class | Val F1 |
|---|---:|
| abnormal | 0.968 |
| artifact | 0.527 |
| normal | 0.704 |

This is the proof-of-concept for the merge. On the *same* underlying recordings that produced macro-F1 0.51–0.59 under five classes, the three-class framing reaches **0.733**, a jump of roughly +0.14 to +0.22 from the taxonomy change alone, before any new data. The *abnormal* class is now strong (0.968) because it is no longer three thin classes but one class with pooled support. *Artifact* (0.527) is still the weak point, foreshadowing that artifact scarcity is a separate, unsolved problem that the taxonomy merge does not touch (artifact was never merged; it stays at 40 samples). The comparatively low accuracy (0.658) against a high macro-F1 (0.733) reflects the small, still-imbalanced corpus at this stage.

### 4.2.3 Run `sfqx9n2b` — the best operating point

With the three-class taxonomy fixed and the PhysioNet expansion folded in (final CardionixDataset, 3,825 records), run `sfqx9n2b` (2025-06-24) is the best operating point of the project. Label space *healthy / abnormal / artifact*, split 0.7 / 0.3.

| Metric | Train | Validation |
|---|---:|---:|
| macro-F1 | 0.802 | **0.863** |
| macro precision | — | 0.870 |
| macro recall | — | 0.863 |
| accuracy | 0.854 | 0.875 |
| weighted-F1 | — | 0.871 |
| loss | 0.324 | 0.284 |

Per-class validation metrics:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| abnormal | 0.929 | 1.000 | 0.963 |
| healthy | 0.896 | 0.945 | 0.920 |
| artifact | 0.784 | 0.643 | **0.707** |

Reading the run:

- **macro-F1 0.863** against the five-class runs' 0.512–0.593 is a recovery of roughly **+0.27** at the top end. This is the single largest improvement in the project, and its source is the label space, not the network.
- **abnormal recall is 1.000.** Every abnormal recording in the validation fold is caught. For a screening tool this is the sign you want on the class you least want to miss, though on a small fold it should be read as "no misses observed here," not "no misses ever." Its precision (0.929) means a modest number of healthy recordings are flagged abnormal, an acceptable trade for a triage instrument that errs toward referral.
- **artifact is still the hardest class** (F1 0.707), and specifically its **recall is low (0.643)**. The model lets roughly a third of unusable recordings pass as if they were valid heart sounds. This is the exact failure mode the field-recording effort (README §5.2) was meant to attack, and it is capped by the same fact all along: artifact has only 40 labelled examples and was never merged with anything. Artifact volume is therefore the clearest data priority coming out of this run.
- **val (0.863) > train (0.802).** As flagged in §4.0, this is the small-validation-fold signature, not evidence of superior generalisation. The gap is small and the losses are close (val 0.284, train 0.324), so there is no sign of overfitting; but the validation macro-F1 should be read with a confidence interval that the fold size does not let us tighten. It is a strong operating point, not a claim of 0.863 on a held-out clinical population.

<div align="center">
<img src="assets/fig_per_class_f1.png" width="720" alt="Per-class precision/recall/F1 of the best model."/>
</div>

### 4.2.4 Best-run configuration

The configuration is recorded here in full because the best result was reached with several *non-obvious* choices (augmentation off, no class weighting), and reproducibility requires them.

| Group | Setting | Value |
|---|---|---|
| Representation | extractor | MFCC |
| | `n_mfcc` | 128 |
| | `n_mels` | 128 |
| | `n_fft` | 2048 |
| | window | 2048 |
| | hop | 1024 |
| Signal | sample rate | 2000 Hz |
| | duration | 10 s (pad/clip) |
| | augmentation | **OFF** (`augment_kwargs = None`) |
| | scaler | none |
| Optimisation | optimiser | Adam |
| | learning rate | 1e-4 |
| | scheduler | ReduceLROnPlateau, patience 5, monitor val/loss |
| | loss | CrossEntropy, weights **[1, 1, 1]** (no class weighting) |
| | batch size | 20 |
| | max / min epochs | 15 / 10 |
| | seed | 42 |
| Selection | checkpoint metric | val weighted-avg F1, `save_top_k = 15` |
| | early stop | patience 10 on val/loss |
| Metrics | reporter | sklearn `classification_report`, $F_\beta$ with $\beta = 1.10$ |

Two of these deserve comment because they run against instinct:

- **Augmentation is off.** The domain-specific augmentation module (time-stretch, amplitude scaling, pitch shift, Gaussian noise, same-class mixing, HPSS) exists and is described in README §7.2, yet the best operating point used none of it. On a small corpus one expects augmentation to help; that it did not is itself a data-side signal, and it is one of the observations that pointed toward the Phase C diagnosis.
- **No class weighting.** The cross-entropy weights are `[1, 1, 1]` despite the corpus being dominated by *healthy/normal* (2,926 of 3,825). The taxonomy merge, not a loss reweighting, is what carried the imbalance. Reweighting was left as a lever for a future round.

### 4.2.5 Per-epoch trajectory of the best run

The checkpoint filenames of `sfqx9n2b` encode the validation macro precision / recall / F1 at each saved epoch. Read in order they show how the operating point was reached, and they expose a precision–recall oscillation that a single final number hides.

| Epoch | Macro precision | Macro recall | Macro F1 | Note |
|---:|---:|---:|---:|---|
| 0 | 0.84 | 0.76 | 0.80 | precision already high, recall lagging |
| 1 | 0.90 | 0.65 | 0.73 | precision up, recall drops hard — conservative model |
| 2 | 0.88 | 0.78 | 0.82 | recall recovers |
| 3 | 0.83 | 0.82 | 0.82 | precision and recall cross, balanced |
| 4 | 0.75 | 0.84 | 0.78 | recall now leads, precision dips — permissive model |
| 5 | 0.87 | 0.82 | 0.85 | both high simultaneously |
| 6 | 0.91 | 0.76 | 0.81 | precision peak, recall dips again |
| 7 | 0.79 | 0.84 | 0.81 | swings back to recall-leading |
| 8 | 0.88 | 0.79 | 0.83 | |
| 9 | 0.89 | 0.83 | **0.86** | **best**: precision and recall both high, closest to balanced-and-high |
| 10 | 0.89 | 0.79 | 0.84 | recall slips |
| 11 | 0.90 | 0.79 | 0.84 | precision-leading again |

<div align="center">
<img src="assets/fig_training_curve.png" width="720" alt="Validation macro precision, recall, F1 per epoch for the best run."/>
</div>

**The oscillation, and what it means.** Precision and recall do not converge smoothly; they alternate leadership from epoch to epoch. Epoch 1 is a *conservative* operating point (precision 0.90, recall 0.65): the model has moved its decision boundary so that it predicts a class only when confident, buying precision at the cost of recall. Epoch 4 is the mirror image, a *permissive* operating point (precision 0.75, recall 0.84): the boundary has moved the other way. Epoch 6 swings back to a precision peak (0.91) with recall sagging to 0.76. This see-saw is characteristic of training on a small, imbalanced validation fold: small shifts in the decision surface between epochs move samples across the boundary, and because each class fold is small, a few reclassified samples swing precision and recall in opposite directions.

The consequence for model selection is concrete. Selecting on precision alone would have picked epoch 6 (0.91) and shipped a model that misses a quarter of positives. Selecting on recall alone would have picked epoch 4 or 7 (0.84) and shipped an imprecise one. **macro-F1 is the right selection metric precisely because it penalises both failure modes**, and it correctly identifies epoch 9 (0.89 / 0.83 → F1 0.86) as the point where precision and recall are simultaneously high rather than one bought at the other's expense. The oscillation is also a reminder that the ±0.02–0.03 epoch-to-epoch jitter in macro-F1 is within the noise band of this validation fold, which is a further reason not to over-read the exact 0.863.

---

## 4.3 Phase C — the plateau and the diagnosis

### 4.3.1 The plateau

With the taxonomy fixed and `sfqx9n2b` established, the natural next move was model-centric: deepen the network, widen the recurrent layers, try the residual-recurrent V2, tune hyperparameters. These changes **stopped producing gains**. The learning curves plateaued regardless of network depth or recurrent width, and, as noted in §4.2.4, the domain-specific augmentation that should have helped a small corpus did not move the best operating point.

### 4.3.2 The diagnosis

The interpretation is standard but it is the hinge of the whole project. When the model class is already expressive enough to fit the available signal, adding capacity does nothing: the loss is not capacity-bound, it is data-bound and representation-bound. Three observations converge on this reading:

1. **Architecture changes did not move the ceiling.** If capacity were the constraint, deeper or wider models would have improved macro-F1; they did not.
2. **Augmentation did not help.** If the constraint were variance in the training distribution, synthetic augmentation would have helped; it did not, which points at the *quality and separability* of the representation rather than its quantity.
3. **The residual weak class is a data class.** *Artifact* (F1 0.707, recall 0.643) is limited by having 40 examples, not by model expressiveness. No architecture recovers a class the data does not describe.

The binding constraint is therefore the **volume and quality of the data and the separability of the representation**, not the capacity of the model. This is the diagnosis that redirected the project from model-centric to data-centric, and it produced two concrete lines of work: the **PhysioNet expansion** (README §5.3, the 585 → 3,825 growth already reflected in `sfqx9n2b`) and the **`cardiobit`** signal-processing library that attacks representation separability directly (README §11).

The order matters for the honesty of the story: `sfqx9n2b` already contains the PhysioNet data, so the data-centric pivot was partly *cause* of the best result and partly *consequence* of the plateau observed around it. The lesson generalises past this corpus: under a data-constrained regime, improving the data and the front end is a more effective lever than adding model capacity.

---

## 4.4 An instrumentation failure we are not hiding

The logged multi-class ROC-AUC reads a constant **0.4000** on **every run, in both training and validation**, across the entire experiment log. It does not move with model quality, with the taxonomy change, or with the +0.27 macro-F1 improvement between Phase A and Phase B.

A metric that is invariant to a change that demonstrably improves the classifier is not measuring the classifier. The value is fixed by a bug in the one-versus-one multi-class AUC logging path (a default aggregation that collapses to a constant), not by any property of the models. In particular, **0.4000 is not a real "below-chance" AUC**; it is a non-value.

We therefore **do not report ROC-AUC as a result anywhere in this project.** It is listed here only so that anyone reading the raw W&B logs is not misled by the constant, and so that fixing the one-versus-one AUC path is on the record as a required instrumentation repair before the next round of experiments. Related and equally honest: probability **calibration** was not evaluated at all, which for a screening instrument whose output feeds a referral decision is a real gap (README §12), not an oversight to be papered over.

---

## 4.5 Summary of runs

| Run | Phase | Taxonomy | Corpus | Val macro-F1 | Val acc | Val loss | Verdict |
|---|---|---|---|---:|---:|---:|---|
| `uutty3vj` | A | 5-class | DHD | 0.593 | 0.744 | 0.711 | minority classes collapse (extrastole 0.22, extrahls 0.29) |
| `qjaxj05j` | A | 5-class | DHD | 0.512 | 0.701 | 0.820 | worse: extrastole F1 = 0.000 |
| `uf7xi6nb` | B | 3-class | pre-PhysioNet | 0.733 | 0.658 | — | merge works even before new data |
| `sfqx9n2b` | B | 3-class | CardionixDataset (3,825) | **0.863** | 0.875 | 0.284 | **best operating point** |

The through-line in one sentence: the largest single gain in the project (+0.27 macro-F1) came from **reframing five over-fragmented labels into three clinically meaningful ones**, not from a bigger network; and once that was done, the evidence pointed at data and representation, not model capacity, as the next constraint to attack.
