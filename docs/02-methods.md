# Appendix 2 — Experimental framework, preprocessing, and augmentation

> Companion to the [README](../README.md). This appendix specifies the configuration surface of the experimental framework, the full ETL and feature-extraction path, the domain-specific augmentation module, and the optimisation and tabular-preprocessing settings. It expands [§6](../README.md#6-experimental-framework) and [§7](../README.md#7-signal-representation-and-preprocessing) of the report. All numbers reproduce the best operating point (three-class, W&B run `sfqx9n2b`); no value here is invented.

---

## A2.1 The configuration-driven framework

The framework (internally packaged as `cardionix`) is built on **PyTorch Lightning** for the training loop and **Weights & Biases** for run tracking (W&B project **"Cardio Sonix"**). Its organising principle is that an experiment is *data*, not code: a run is fully described by a set of typed configuration objects, and every hypothesis (a taxonomy, a feature extractor, an augmentation schedule, an architecture) is expressed as a change to those objects rather than an edit to the training script.

### A2.1.1 Pydantic-typed configuration models

Four **Pydantic** models carry the complete run specification. Each is a typed, validated container; construction fails loudly if an argument is missing, mistyped, or out of range.

| Config model | Governs | Representative fields |
|---|---|---|
| `ClassifyDatasetParams` | corpus selection, label taxonomy, class merges/exclusions, source annotation | dataset name, class-merge rule, taxonomy mode (3-class / 5-class), source filter |
| `ETLPipelineParams` | the audio and tabular preprocessing chain | resample rate, target duration, pad/clip policy, pad mode, extractor spec, scaler spec |
| `DataModuleParams` | splitting, batching, augmentation schedule, workers | split ratio, `batch_size`, `augment_kwargs`, `num_workers`, seed |
| `LightningModuleParams` | model, optimiser, loss, scheduler, metric configuration | architecture, `lr`, loss weights, scheduler patience, monitored metric |

Typing the configuration buys three things directly.

1. **Input validation at construction.** Every argument is checked before a single batch is loaded. A malformed extractor spec or an out-of-range split ratio raises at object construction, not three epochs into a GPU run. This is the difference between a typo costing seconds and costing hours.
2. **Auto-logged full run spec.** Because the configuration is a serialisable typed object, the *entire* specification of a run (down to `hop_length` and the pad mode) is logged to W&B automatically. A result in "Cardio Sonix" can always be traced back to the exact configuration that produced it; there is no gap between "what we ran" and "what we recorded."
3. **Taxonomy, extractor, and augmentation swapped via config, not code.** Moving from the five-class to the three-class taxonomy, replacing MFCC with a mel-spectrogram, or toggling the augmentation pipeline is a field change in `ClassifyDatasetParams`, `ETLPipelineParams`, or `DataModuleParams`. The training code is untouched, so a systematic sweep over hypotheses is a sweep over configuration values, and every point in that sweep is reproducible from its logged spec.

The practical consequence: the taxonomy-collapse experiment that recovered +0.27 macro-F1 (README [§10.2](../README.md#102-collapsing-the-taxonomy-recovered-027-macro-f1)) was a change to the class-merge rule in `ClassifyDatasetParams`, not a rewrite of the data loader.

---

## A2.2 The ETL pipeline

The ETL layer transforms raw, heterogeneous recordings into fixed-shape tensors ready for the model. It handles two modalities (audio and tabular) through separate preprocessors composed under `ETLPipelineParams`.

### A2.2.1 Audio path

The audio preprocessor executes an ordered chain:

1. **Load and channel reduction.** Read the waveform and reduce to **mono** (average of channels), so that device-specific stereo layouts do not leak into the representation.
2. **Resampling.** Resample every recording to a single canonical rate. The corpus is genuinely multi-rate (2000 Hz for the PhysioNet portion, 4000 Hz for DigiScope, 44100 Hz for the iStethoscope portion), so this step is what makes a merged corpus trainable at all. The best runs resample to **2000 Hz**, matched to the 20–200 Hz diagnostic band and to the PhysioNet convention; by the Nyquist criterion, 2000 Hz preserves content up to 1000 Hz, comfortably above the useful cardiac band.
3. **Duration standardisation.** Recordings vary from under a second to several minutes (DHD-era statistics: mean 9.07 s, median 7.39 s, min 0.76 s, max 210 s). The pipeline forces a **fixed duration** $D$ (best run: $D = 10$ s, i.e. $N = D \cdot f_s = 20{,}000$ samples at 2000 Hz):
   - signals **longer** than $D$ are **clipped** to the first $N$ samples;
   - signals **shorter** than $D$ are **padded** to $N$ samples under a configurable **pad mode**.
4. **Feature extraction.** The standardised waveform is passed to the configured extractor (MFCC in the best run; see [A2.3](#a23-feature-extraction-mfcc)).
5. **Augmentation (train only).** If enabled, augmentation is applied *after* standardisation and *only* on the training partition (see [A2.2.3](#a223-augment-on-train-only-locked-on-valtest)).

**Pad modes.** For a short clip of length $L < N$, the pipeline supports the following strategies for synthesising the missing $N - L$ samples:

| Pad mode | Fill behaviour | Character for PCG |
|---|---|---|
| `constant` | fill with a constant (zero) | introduces a silent tail, no spurious spectral content; **used in the best run** |
| `edge` | repeat the boundary sample | holds a DC-like level, can bias the envelope |
| `reflect` | mirror the signal about the edge (endpoint not repeated) | preserves local spectral statistics, no discontinuity |
| `symmetric` | mirror including the edge sample | as `reflect`, with the boundary sample duplicated |
| `nearest_neighbors` | extend using neighbouring content | fills with plausible cardiac-cycle content rather than silence |

The best run uses `pad_mode = constant`: zero-padding a short recording adds no synthetic heartbeats and no spurious spectral energy, which is the conservative choice when the missing content is genuinely unknown.

### A2.2.2 Class handling at load time

Two label operations happen inside the ETL layer, driven by `ClassifyDatasetParams`, so that no model code depends on the taxonomy.

- **Class-merge at load.** The raw PASCAL labels are folded into the operational three-class taxonomy by a configurable rule, `abnormal = {murmur, extrahls, extrastole}`, `healthy = normal`, `artifact = artifact`. This merge is applied when annotations are read, so switching between the five-class and three-class views is a configuration change, and it is the mechanism behind the central experimental result of the report.
- **Neighbor-merge for short clips.** Recordings too short to carry a full standardised window can be concatenated with an adjacent same-source clip (a "neighbor merge") rather than being padded from nothing, recovering usable duration from fragmentary recordings before the pad/clip step decides the final length.

### A2.2.3 Augment on train only, locked on val/test

Augmentation is a property of the **training partition alone**. The `DataModuleParams.augment_kwargs` schedule is applied to training batches; validation and test partitions are served **without** augmentation, so the reported metrics measure performance on the true signal distribution, not on a widened one. This locking is enforced at the DataModule level, not left to convention, which removes a common and silent source of optimistic bias in audio pipelines.

---

## A2.3 Feature extraction (MFCC)

The primary representation is the **Mel-Frequency Cepstral Coefficient** sequence. MFCCs compress the short-time spectrum onto a perceptually spaced filterbank and then decorrelate the log-band energies, producing a compact time-frequency description well suited to the band-limited, quasi-periodic PCG signal.

### A2.3.1 Mel scale

The mel scale approximates the nonlinear frequency resolution of the cochlea, spacing filters more densely at low frequencies (exactly where S1/S2 and murmurs live):

$$
m(f) = 2595 \, \log_{10}\!\left(1 + \frac{f}{700}\right).
$$

The best run uses the **HTK** mel formulation (the single-log form above, as opposed to the piecewise Slaney convention), set by `htk = True`.

### A2.3.2 Mel filterbank energies

The waveform is framed (window length `win_length`, hop `hop_length`) and each frame is transformed by the discrete Fourier transform of size `n_fft`. Let $|X_t(f)|^2$ be the power spectrum of frame $t$. A bank of $B$ triangular filters $\{H_b(f)\}_{b=1}^{B}$, spaced uniformly on the mel axis $m(f)$ between a lower and an upper edge frequency, integrates the power spectrum into $B$ **mel-band energies**:

$$
E_b(t) = \sum_{f} H_b(f)\,\lvert X_t(f)\rvert^2,
\qquad b = 1, \dots, B .
$$

Each triangular filter $H_b$ peaks at its centre mel frequency and falls linearly to zero at the centres of its neighbours, so the bank forms a smooth, overlapping partition of the spectrum on the mel axis.

### A2.3.3 Discrete cosine transform to cepstral coefficients

The log mel-band energies are decorrelated by the type-II **discrete cosine transform**, yielding the cepstral coefficients:

$$
c_k(t) = \sum_{b=1}^{B} \log\!\big(E_b(t)\big)\,
\cos\!\Big[\frac{\pi k}{B}\Big(b - \tfrac{1}{2}\Big)\Big],
\qquad k = 1, \dots, K .
$$

The DCT concentrates the spectral-envelope information in the low-order coefficients and approximately whitens the feature vector (its basis is close to the Karhunen–Loève transform of log-filterbank energies), which is why MFCCs are a compact, largely decorrelated front end. The log compresses the wide dynamic range of the band energies before decorrelation.

### A2.3.4 Best-run parameters and output shape

| Parameter | Value | Role |
|---|---|---|
| `n_mfcc` ($K$) | 128 | number of cepstral coefficients retained |
| `n_mels` ($B$) | 128 | number of mel filterbank channels |
| `n_fft` | 2048 | DFT size per frame |
| `win_length` | 2048 | analysis window length (samples) |
| `hop_length` | 1024 | frame advance (samples), i.e. 50% overlap |
| mel scale | `htk` | HTK single-log mel formulation |
| `sample_rate` | 2000 Hz | canonical resample rate |
| `duration` | 10 s | standardised clip length ($N = 20{,}000$ samples) |
| channels | mono | single-channel input |
| `pad_mode` | `constant` | zero-padding for short clips |
| scaler | none | no post-extraction feature scaling in the best run |

**Output shape $(K \times T)$.** With $K = 128$ cepstral coefficients and a number of frames

$$
T \;\approx\; \left\lfloor \frac{N - \texttt{win\_length}}{\texttt{hop\_length}} \right\rfloor + 1 ,
$$

the extractor emits a $(K \times T)$ time-frequency matrix per recording: 128 coefficient rows against $T$ time frames. For $N = 20{,}000$, `win_length` $= 2048$, `hop_length` $= 1024$, this gives $T \approx 18$ frames. That $(128 \times T)$ matrix is the tensor consumed by the recurrent and residual encoders of the CardioNet family (README [§8](../README.md#8-model-architectures-the-cardionet-family)): the $K$ axis is treated as the feature dimension and the $T$ axis as the sequence over which the LSTM integrates the systole–diastole rhythm.

---

## A2.4 Domain-specific augmentation (`AudioAugmentations`)

Because the corpus is small and the signal is quasi-periodic, the augmentation module is tuned to phonocardiography rather than borrowed wholesale from speech or general audio ML. Each transform is chosen so that it widens the corpus's variance **without destroying diagnostic content** (the S1/S2 timing and the murmur texture between them). All transforms are applied on the training partition only.

| Transform | Range / coefficient | Physical meaning |
|---|---|---|
| `time_stretch` | rate 0.5–1.8 | change tempo without pitch |
| `value_augment` (amplitude scaling) | scale 0.5–3.0 | change loudness |
| `pitch_shift` | ±5 semitones | change pitch without tempo |
| `gaussian_noise` | coeff $0.005 \cdot \max\lvert x\rvert$ | additive white noise |
| `noise_by` (same-class mixing) | coeff 0.05–0.3 | blend with another same-class sample |
| `hpss` | random remix | harmonic–percussive re-proportioning |

**Why each is domain-appropriate for quasi-periodic heart sounds.**

- **`time_stretch` (0.5–1.8).** Stretching the time axis is a direct model of **heart-rate variation**. A heart at 50 bpm and at 90 bpm produces the same S1/S2 morphology at different cadences; time-stretching manufactures that variation from a single recording without altering the spectral signature of a valve event. This is the single most physically motivated PCG augmentation.
- **`value_augment` / amplitude scaling (0.5–3.0).** Smartphone capture level depends on contact pressure, pickup point, and gain, so the *absolute* amplitude of a recording carries no diagnostic information. Scaling amplitude teaches the model **invariance to capture loudness**, which is exactly the nuisance variable that varies most across devices and users.
- **`pitch_shift` (±5 semitones).** Modest pitch shifts model **inter-subject and inter-device spectral shift**: tissue transfer functions and microphone responses shift the dominant frequency band of S1/S2 and murmurs from person to person. A bounded shift widens the spectral distribution the model must tolerate while staying inside the plausible cardiac band.
- **`gaussian_noise` ($0.005 \cdot \max\lvert x\rvert$).** The coefficient is scaled to each recording's own peak amplitude, so the injected noise floor is a *fixed fraction* of the signal rather than an absolute level. This models low-level microphone and electronic noise and discourages the model from keying on the pristine silence between beats.
- **`noise_by` (same-class mixing, 0.05–0.3).** Blending a recording with a **different sample of the same class** at a small coefficient is a PCG-specific mixup: it synthesises new within-class variance (a second person's murmur texture layered under the first) without crossing a decision boundary, because the mixing partner shares the label. It widens the class manifold rather than blurring the classes together.
- **`hpss` (harmonic–percussive source separation, random remix).** HPSS decomposes the signal into a **harmonic** component (sustained, tonal: the murmur texture and flow sounds) and a **percussive** component (transient, broadband: the S1/S2 valve closures). Randomly re-proportioning the two before recombining lets the model see the *transients* and the *texture* in varying relative strength, which is precisely the axis along which a soft murmur differs from a clean beat. It is a decomposition that maps onto the physics of the two heart-sound families.

**An honest note on the best run.** The single best operating point (run `sfqx9n2b`, macro-F1 0.863) was reached with augmentation **disabled** (`augment_kwargs = None`). We report this rather than hide it, because it is itself evidence about the binding constraint. If regularisation were the limiting factor, augmentation would have helped; that it did not is consistent with the report's central diagnosis (README [§10.3](../README.md#103-the-diagnosis-model-capacity-was-not-the-bottleneck)) that the bottleneck was the **volume and representation of the data**, not overfitting. Augmentation widens variance around existing samples; it cannot manufacture the statistical weight that the PhysioNet expansion supplied. The negative result points at the same conclusion as the positive one.

---

## A2.5 Optimisation

The optimiser, scheduler, loss, and training-control settings for the best run are collected below, governed by `LightningModuleParams` and `DataModuleParams`.

| Setting | Value | Notes |
|---|---|---|
| Optimiser | Adam, `lr` $= 10^{-4}$ | |
| LR scheduler | `ReduceLROnPlateau`, patience 5 | monitors `val/loss` |
| Loss | `CrossEntropyLoss`, weights `[1, 1, 1]` | **uniform**, no imbalance weighting |
| Batch size | 20 | |
| `max_epochs` | 15 | |
| `min_epochs` | 10 | |
| Early stopping | patience 10 | monitors `val/loss` |
| Seed | 42 | |
| `num_workers` | 12 | |

The optimisation objective is the weighted cross-entropy

$$
\mathcal{L} = -\sum_{c=1}^{C} w_c \, y_c \log \hat{p}_c ,
$$

with per-class weights $w_c$. Adam at $10^{-4}$ is a deliberately conservative learning rate for a small corpus; `ReduceLROnPlateau` on `val/loss` (patience 5) lets it decay once validation loss stalls, and early stopping on the same signal (patience 10), bounded below by `min_epochs = 10` and above by `max_epochs = 15`, caps the run. The seed is fixed at 42 for reproducibility, and `num_workers = 12` saturates the data-loading path on this machine.

**The uniform class weights are a design choice worth revisiting.** The loss weights are $[1, 1, 1]$: every class contributes equally per-sample, with **no reweighting for imbalance**, despite a genuinely imbalanced three-class corpus (normal 2926, abnormal 859, artifact 40). This is the conservative default, and it did not prevent the headline result, but it is very likely suboptimal for the *artifact* class specifically. Artifact is both the scarcest class (40 labelled samples) and the hardest (validation F1 0.707, recall 0.643, README [§10.2](../README.md#102-collapsing-the-taxonomy-recovered-027-macro-f1)); its low recall is exactly the symptom that inverse-frequency or effective-number class weighting is meant to address. Because the checkpoint objective is **macro**-F1, which already refuses to let the majority class dominate the *selection* signal, the effect of uniform weights is felt in the *gradient* rather than in model selection: the loss still under-weights the minority class during optimisation even though the model is chosen on a balanced metric. Introducing imbalance-aware weights (or a focal loss) is a low-cost change worth testing before the next round; the framework supports it as a single edit to the loss-weights field in `LightningModuleParams`.

---

## A2.6 Tabular preprocessing (V2 multimodal experiment)

The tabular branch supports the **CardioNet V2** multimodal experiment (README [§8.2](../README.md#82-cardionet-v2-a-multimodal-residual-recurrent-experiment)). It is a scikit-learn preprocessing chain applied to questionnaire-style features, fitted on the training partition and then reused verbatim on validation and test.

| Feature type | Transform | Purpose |
|---|---|---|
| categorical | `OneHotEncoder` | expand nominal categories into indicator columns |
| numerical | `StandardScaler` | zero-mean, unit-variance scaling |
| numerical | `Normalizer` | per-sample vector normalisation to unit norm |
| numerical | `MinMaxScaler` | scale to a fixed `[0, 1]` range |

The numerical transform is selectable (`StandardScaler` / `Normalizer` / `MinMaxScaler`) via configuration, and the fitted preprocessor is **cached with `joblib`** so that the exact transform learned on the training fold is serialised, reloaded, and applied identically at inference. Caching the fitted object (not just its parameters) removes any risk of a train/inference skew in the tabular path.

**The same caveat as the report.** The tabular source is an external population survey (a CDC/BRFSS-style questionnaire); its respondents are **not** the same individuals as the PCG recordings, so the audio–table pairs are not per-subject aligned. CardioNet V2 is therefore an **architectural experiment in multimodal fusion, not a validated multimodal diagnostic model**, and the best reported operating point is audio-only (`extra_filepath = None`). The tabular preprocessing is documented here for completeness of the framework, not as a component of any headline result.
