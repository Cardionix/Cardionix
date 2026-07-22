# Appendix 1 — The corpus

> Companion to [`../README.md`](../README.md), §5 (*Data: building the corpus*). This appendix records the full detail behind the corpus: the repair of the source data, the metadata schema and its provenance rationale, the three corpus generations with exact counts, the field-recording effort that extended the artifact class, the class-merge rule, the split protocol, and the one limitation (patient-level splitting) that conditions every reported metric. All numbers here are the real numbers of the project; none are invented for exposition.

---

## A1.1 The corpus was the hard part

A survey of open sources (Kaggle, Google Dataset Search, and the phonocardiography literature) confirmed a structural scarcity of usable heart-sound corpora. Heart-sound audio is expensive to collect, ethically encumbered, and rarely released with clean labels. The one public dataset we could realistically build on, derived from the PASCAL Classifying Heart Sounds Challenge and redistributed through Kaggle, arrived in a state that was not directly usable for supervised learning. The bulk of the early project effort was therefore not modelling but data engineering, and it mattered more to the final result than any single architectural change.

Four defects had to be repaired before a single model could be trained honestly.

1. **Broken label-to-audio mapping.** The correspondence between the label manifests and the audio filenames had been lost in redistribution. A recording without a trustworthy label is worse than no recording: it silently corrupts both training and evaluation. Recovering the mapping was the precondition for everything else.
2. **Heterogeneous formats and sample rates.** The audio came from two different capture devices at two different sample rates (44100 Hz and 4000 Hz), in inconsistent containers and channel layouts. A learner that ingests such a mixture without harmonisation is partly learning the device, not the physiology.
3. **Severe class imbalance.** The labelled classes ranged from 351 recordings (*normal*) down to 19 (*extra heart sound*), a spread wide enough to let a majority-class predictor score deceptively well while learning nothing useful about the rare classes. With $n_{\max} = 351$ and $n_{\min} = 19$ the raw imbalance ratio was
   $$
   \rho = \frac{n_{\max}}{n_{\min}} = \frac{351}{19} \approx 18.5,
   $$
   which is the quantitative root of the five-class collapse documented in the README (§10.1).
4. **No structured metadata.** There was no per-recording record of duration, sample rate, capture device, or source. Without such a record, a multi-source merge is unauditable: once files from different corpora are pooled, one can no longer ask *where did this recording come from* or *at what rate was it captured*, and any later analysis of device- or rate-dependent effects is impossible.

### The repair

We rebuilt the corpus from the ground up rather than patch it in place:

- **Recovered the label-to-audio mapping** that the original release had lost, restoring a trustworthy label for every retained recording.
- **Standardised every file** to a single container and a canonical sample rate, converting all audio to **mono**, so that channel layout and rate were no longer confounded with the label.
- **Adopted a UUID naming regime.** Each recording was renamed to a universally unique identifier, decoupling the file's identity from its original (inconsistent, sometimes label-leaking) filename and eliminating collisions across sources.
- **Reorganised the storage layout** into a consistent, source-partitioned tree with a single canonical annotation manifest.
- **Enriched every recording with metadata** (schema below), so that provenance travels with the audio and later merges remain auditable.

The cleaned artifact of this repair is published as the **Dangerous Heartbeat Dataset (DHD)**. DHD is the corrected, standardised, metadata-carrying version of the PASCAL/Kaggle material; it is the first of the three corpus generations described in §A1.3.

---

## A1.2 The metadata schema

Every recording in the canonical annotation carries seven fields:

| Column | Type | Meaning |
|---|---|---|
| `filename` | string (UUID) | canonical UUID identifier of the standardised audio file |
| `label` | categorical | class label (`normal`, `murmur`, `extrastole`, `extrahls`, `artifact`; binary `normal`/`abnormal` for PhysioNet-origin rows) |
| `duration` | float (s) | length of the standardised recording in seconds |
| `sr` | int (Hz) | sample rate of the recording (2000, 4000, or 44100) |
| `device` | string | capture device (e.g. iStethoscope Pro iPhone app, DigiScope digital stethoscope) |
| `source` | string | originating corpus (Kaggle/PASCAL, PhysioNet/CinC 2016) |
| `date` | date | acquisition or ingestion date |

The three provenance columns (`device`, `source`, `sr`) are the load-bearing part of the schema. The central engineering problem of this project is that the final corpus is a **merge of heterogeneous sources** captured on different instruments at different sample rates. Without provenance carried per recording, such a merge is a black box: one cannot verify that a reported class distribution matches its sources, cannot isolate a device- or rate-specific artifact, cannot reconstruct which generation a recording belongs to, and cannot audit whether an evaluation fold accidentally over-represents one source. Because provenance is attached at ingestion and survives the merge, every downstream claim about the corpus (the counts in §A1.3, the sample-rate mix, the source breakdown) is reproducible from the manifest rather than asserted. In a research context where the merge is itself a contribution, an auditable merge is the difference between a defensible corpus and an opaque one.

`sr` additionally functions as an experimental control. Because the diagnostic energy of heart sounds occupies roughly the 20–200 Hz band (README §3), all rates are ultimately resampled to a canonical training rate (2000 Hz in the best runs), but retaining the *original* rate in metadata lets us reason about what information each source could have carried before resampling: a 4000 Hz DigiScope recording and a 44100 Hz phone recording reach the model at the same rate, yet their pre-resampling ceilings differ, and only the metadata records that.

---

## A1.3 The three corpus generations

The corpus evolved through three generations. The first repaired and standardised the PASCAL/Kaggle material; the second added statistical weight through PhysioNet; the third merged and harmonised the two into the final training corpus.

### Generation 1 — DHDataset (DHD)

DHD is the repaired PASCAL/Kaggle corpus: **585 labelled** recordings plus **247 unlabelled** recordings held in reserve. Its five-class label distribution is:

| Class | Count | Share of labelled |
|---|---:|---:|
| normal | 351 | 60.0% |
| murmur | 129 | 22.1% |
| extrastole | 46 | 7.9% |
| extrahls (extra heart sound) | 19 | 3.2% |
| artifact | 40 | 6.8% |
| **total labelled** | **585** | 100% |

The material derives from the Kaggle `kinguistics/heartbeat-sounds` release (the PASCAL Classifying Heart Sounds Challenge material, Bentley et al.), captured on two devices:

| Device | Original sample rate | Files |
|---|---:|---:|
| iStethoscope Pro (iPhone app) | 44100 Hz | 124 |
| DigiScope (digital stethoscope) | 4000 Hz | 461 |
| **total** | | **585** |

The two-device split is exactly why the `device`/`sr` provenance columns matter: 124 recordings arrive at consumer-phone rate and 461 at digital-stethoscope rate, and only the metadata preserves that distinction after standardisation.

Duration statistics in the DHD era, computed over the standardised recordings, are:

| Statistic | Value (s) |
|---|---:|
| mean | 9.07 |
| median | 7.39 |
| minimum | 0.76 |
| maximum | 210.00 |

The right-skew (mean above median, a 210 s maximum against a 0.76 s minimum) motivates the fixed-duration standardisation used downstream (pad or clip to a fixed 10 s window, README §7.1) and the neighbour-merge handling of very short clips in the ETL pipeline.

### Generation 2 — PhysioNetDataset

The second generation is the **PhysioNet/CinC Challenge 2016** database, the largest public PCG corpus, aggregated from nine sources and labelled binary *normal* / *abnormal*:

| Class | Count |
|---|---:|
| normal | 2575 |
| abnormal | 665 |
| **total** | **3240** |

All PhysioNet recordings are at **2000 Hz**. This generation is the source of the project's statistical weight: it is roughly $3240 / 585 \approx 5.5\times$ the size of DHD on its own, and it is what turned a small experiment into a corpus large enough to support the three-class result.

### Generation 3 — CardionixDataset (final merged corpus)

The final corpus merges and harmonises DHD and PhysioNet:

$$
\text{CardionixDataset} = \text{DHDataset (585)} \;+\; \text{PhysioNetDataset (3240)} \;=\; 3825 .
$$

Its three-class distribution and its sample-rate composition are:

| Class | Count | Share |
|---|---:|---:|
| normal (healthy) | 2926 | 76.5% |
| abnormal | 859 | 22.5% |
| artifact | 40 | 1.0% |
| **total** | **3825** | 100% |

| Original sample rate | Files | Source |
|---|---:|---|
| 2000 Hz | 3240 | PhysioNet/CinC 2016 |
| 4000 Hz | 461 | DigiScope (PASCAL) |
| 44100 Hz | 124 | iStethoscope (PASCAL) |
| **total** | **3825** | |

The `normal` count of 2926 is the sum of DHD normal (351) and PhysioNet normal (2575); the `abnormal` count of 859 is the sum of PhysioNet abnormal (665) and the DHD non-normal, non-artifact classes merged under the rule of §A1.5 (murmur 129 + extrastole 46 + extrahls 19 = 194), giving $665 + 194 = 859$. The `artifact` count (40) is carried unchanged from DHD, since PhysioNet has no artifact label. Every count in the merged corpus therefore reconstructs from the provenance-tagged source rows, which is the auditability property of §A1.2 in action.

### Growth across generations

| Generation | Records | Classes present |
|---|---:|---|
| DHDataset | 585 (+247 unlabelled) | 5-class (normal, murmur, extrastole, extrahls, artifact) |
| PhysioNetDataset | 3240 | 2-class (normal, abnormal) |
| CardionixDataset | 3825 | 3-class (healthy, abnormal, artifact) |

The corpus grew by a factor of
$$
\frac{3825}{585} \approx 6.5 ,
$$
which is the data-centric pivot expressed as a single number. This growth is not cosmetic: it changed which experiments were worth running. Under 585 recordings fragmented across five classes, the rare labels were statistically starved; at 3825 recordings across three classes the abnormal and healthy classes are populous enough to train and evaluate credibly, and the residual scarcity is concentrated where it is hardest to remove (the artifact class), which is precisely the target of the field-recording effort below.

---

## A1.4 Extending the artifact class with field recordings

The *artifact* class comprises recordings in which no heartbeat is identifiable. It is the class that lets a screening application refuse a recording ("this is unusable, try again") rather than silently misclassify noise as physiology. It was simultaneously the scarcest labelled class (40 recordings) and the most important for real-world robustness, because a screening tool that cannot recognise its own failure conditions is not deployable. Rather than treat this noise as material to be discarded, we treated it as a class to be modelled.

We first analysed the operating conditions of the intended application (a phone pressed to clothed skin, indoors and outdoors, in the hands of a non-clinician) and enumerated the acoustic interference typical of them. The noise domains identified were:

- **human speech** (the operator or bystanders talking during capture);
- **breathing** into or near the microphone;
- **cloth and contact friction** at the skin-device interface;
- **ambient room and street noise** (background hum, traffic, indoor reverberation);
- **digital and electronic interference** (clipping, codec artifacts, electrical noise);
- **medical-environment sounds** (equipment, alarms, movement in a clinical setting);
- **animals** (domestic-environment interference);
- **movement and footsteps** (gross motion of the operator or subject).

That taxonomy of noise domains, and the recording checklist it produced, is preserved in [`NOISE_CATEGORY_MINING.md`](NOISE_CATEGORY_MINING.md). Working against that checklist, we collected **field recordings** of two kinds: **isolated noise** (each interference domain captured on its own) and **heavily contaminated phonocardiograms** (real heart-sound attempts overwhelmed by one or more of the noise domains), captured both **indoors and outdoors**. These field recordings were folded into the artifact class of the corpus.

The purpose was twofold. First, to **relieve the class imbalance**: the artifact class was the thin end of the distribution, and enlarging it with authentic negative-space recordings is a more faithful remedy than synthetic oversampling. Second, to **improve representativeness**: a screening application will meet exactly these interference conditions in the field, so an artifact class assembled from the field is a better model of the deployment distribution than any laboratory approximation. The field-recording effort is thus a first-class contribution to the corpus, motivated directly by the "acquisition is part of the model" argument of the README (§3): if the sensor and its noise environment are set at capture time, then the negative class must be sampled from that same environment.

---

## A1.5 The class-merge rule

The operational taxonomy collapses the raw five-class labels into three clinically meaningful classes through a configuration-level rule (no re-labelling of the underlying audio, so the merge is reversible and auditable):

$$
\texttt{abnormal} = \{\texttt{murmur},\ \texttt{extrahls},\ \texttt{extrastole}\}, \qquad
\texttt{healthy} = \texttt{normal}, \qquad
\texttt{artifact} = \texttt{artifact}.
$$

The rule is applied at load time by the class-management mechanism of the framework (README §6), which means the same corpus can be trained under the five-class or the three-class taxonomy by editing configuration rather than data. The clinical justification is that for **triage** purposes a murmur, an extrasystole, and an extra heart sound are all *not-normal*; the distinctions between them, while diagnostically real, are not what a screening instrument needs to resolve and were fragmenting scarce data across labels the tool does not need to separate. The empirical justification is the central experimental result of the project (README §10.2): the merge lifted validation macro-F1 from the 0.51–0.59 range of the five-class runs to **0.863**, the single largest improvement in the work, obtained by reframing the label space rather than by enlarging the network. The `artifact` class is deliberately kept separate from `abnormal`: an unusable recording is an operational outcome (re-record), not a physiological one (refer), and conflating the two would defeat the purpose of the class.

---

## A1.6 Splits

Data are partitioned into training and validation folds. Three split configurations were used:

| Split | Ratios | Purpose |
|---|---|---|
| Primary | 0.7 / 0.3 (train / val) | the default under which the reported best operating point was obtained |
| Alternate | 0.8 / 0.2 (train / val) | a higher-data-fraction variant, to check that results were not an artifact of the 0.7/0.3 fraction |
| Three-way | train / val / held-out test | to detect leakage from repeated validation-based hyperparameter selection |

The three-way split exists to guard against a specific and common failure: when hyperparameters are chosen repeatedly against the same validation fold, the validation score gradually ceases to be an unbiased estimate of generalisation and becomes partly fitted to that fold. Introducing a **held-out test set**, untouched during model and hyperparameter selection, lets us check whether the validation-selected configuration still holds up on data it was never tuned against. This is a check on the *selection* procedure, distinct from and additional to the patient-level limitation described next.

---

## A1.7 The patient-level split limitation

One limitation conditions every metric reported anywhere in this project and must be read before any score is trusted. **Patient-level splitting was not possible.** Neither source corpus (PASCAL/Kaggle nor PhysioNet/CinC 2016) provides subject identifiers linking recordings to the individuals they came from. When a subject contributed more than one recording, those recordings cannot be forced onto the same side of the train/validation boundary, because the information needed to group them does not exist in the source metadata.

The consequence is **information leakage across the split**: two recordings of the same heart, sharing that heart's idiosyncratic acoustic signature, can land one in training and one in validation. The model can then be rewarded at validation time partly for having memorised subject-specific characteristics rather than for having learned the *normal*-versus-*abnormal* distinction that generalises to unseen subjects. Reported scores are therefore **optimistically biased** relative to a clean, subject-disjoint evaluation: the true generalisation performance on genuinely new patients is expected to be somewhat lower than the numbers this project reports.

We state this openly because a screening claim is only as credible as its stated failure modes, and because no post-hoc correction is available: the bias cannot be estimated or removed without the subject linkage that the source data omits. Fixing it structurally requires a **proprietary corpus that preserves the recording-to-subject link**, which would permit correct patient-disjoint (subject-level) splitting and lift the bias. That corpus is the first item of the project's future work (README §13).

---

## A1.8 Summary

The corpus is the substrate on which every result rests, and building it was the larger share of the work. Starting from a single unusable public dataset, we recovered its lost labels, standardised its heterogeneous audio to mono at a canonical rate under a UUID naming regime, attached provenance metadata that made a later multi-source merge auditable, and published the cleaned result as DHD. We then extended the scarce and safety-critical artifact class with field recordings collected against an explicit noise taxonomy, integrated the PhysioNet/CinC 2016 database to grow the corpus roughly 6.5× to 3825 recordings, and collapsed an over-fragmented five-class taxonomy into three clinically meaningful classes. The one limitation we cannot engineer away with the available data, patient-level splitting, is stated plainly and carried forward as the primary target of future work.
