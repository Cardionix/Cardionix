# Appendix 3 — The CardioNet architecture family

> Companion to [`README.md`](../README.md), §8 (*Model architectures: the CardioNet family*).
> This appendix gives the layer-level specification of the three CardioNet
> generations, the rationale for each design, and the mathematics of the
> recurrent, residual, and spectral-image components. Source of record:
> [`cardionix/models/baseline.py`](../cardionix/models/baseline.py),
> [`cardionix/models/cardionetV2.py`](../cardionix/models/cardionetV2.py),
> [`cardionix/models/utils/resnet.py`](../cardionix/models/utils/resnet.py),
> [`cardionix/models/utils/mixer.py`](../cardionix/models/utils/mixer.py),
> and the design note [`CardionetV3.md`](CardionetV3.md).

---

## A3.0 Why not fine-tune a large pretrained audio model

Before describing what we built, we record why we did not take the path that
most audio-classification projects now take by default: fine-tuning a large
model pretrained on AudioSet-scale corpora (Wav2Vec2, HuBERT, PANNs, AST, and
similar). Three arguments, in decreasing order of force.

**(a) Domain shift.** The signal these backbones learned is airborne,
wide-band (nominally 0 to 8 kHz or wider), and semantically rich: speech
phonemes, instruments, environmental events. A smartphone phonocardiogram is
none of those things. As derived in README §3, the captured signal is

$$
x(t) = \big(s * h_{\text{tissue}}\big)(t) + n_{\text{env}}(t) + n_{\text{contact}}(t),
$$

that is, a valvular source $s(t)$ convolved with the low-pass impulse response
$h_{\text{tissue}}$ of skin, fat, muscle, and thoracic wall. The diagnostic
energy lives in a narrow band, roughly 20 to 200 Hz, and the signal is
quasi-periodic (the cardiac cycle repeats) rather than semantically sequenced.
The first-layer filters and the learned statistics of a speech/AudioSet model
are tuned to a distribution that overlaps ours only at the very bottom of its
range. Transfer therefore imports mostly irrelevant structure, and the
inductive bias can actively mislead.

**(b) Corpus size.** Even the final merged corpus is 3,825 recordings (README
§5). Fine-tuning a backbone with tens or hundreds of millions of parameters on
a corpus of this size is a recipe for overfitting, and it does not address the
constraint that actually binds (README §10.3: capacity was not the
bottleneck). A large pretrained model would spend its capacity memorising a
small, leakage-prone corpus rather than learning a transferable representation
of heart sound.

**(c) Research economy.** At this stage the open question was *how the signal
should be represented*, not how many parameters could be thrown at it. A
lightweight model trains in minutes on a single laptop GPU, so a representation
hypothesis (this MFCC configuration, this taxonomy, this front end) can be
tested an order of magnitude more cheaply than one fine-tuning run of a large
backbone. The CardioNet family is deliberately small so that iteration speed,
not compute, sets the pace of the research.

The correct heavy-weight move for this domain is not supervised transfer from
speech, but *self-supervised pretraining on unlabelled PCG* (README §13, item
2). That is future work, not the subject of this appendix.

---

## A3.1 CardioNet V1 — convolutional-recurrent baseline (`BaselineRNNModel`)

V1 is the reference architecture: a learned 1-D filterbank feeding a recurrent
temporal model feeding a linear decoder. Its job is to be expressive enough to
fit the signal while remaining cheap to iterate on.

```
  MFCC sequence  x : (C_in, T)
        │
        ▼
  ┌───────────────────────── Encoder (learned filterbank) ─────────────────────────┐
  │  for each width w in encoder_depth = [2048, 1024, 512]:                          │
  │      Conv1d(k=5, stride=1, padding="same")  →  ReLU  →  MaxPool1d(k=2, s=2)      │
  │      →  BatchNorm1d(w)   [→ Dropout, except final block]                         │
  └─────────────────────────────────────────────────────────────────────────────────┘
        │   (channels = 512, time downsampled by 2 per block)
        ▼
  ┌───────────────────────── LSTMModule (stacked LSTM) ─────────────────────────────┐
  │  LSTM(→256, batch_first)  →  LSTM(256→128, batch_first)                          │
  │  returns final hidden state h_T                                                  │
  └─────────────────────────────────────────────────────────────────────────────────┘
        │   h_T : (128,)
        ▼
  ┌───────────────────────── Decoder (FC head) ─────────────────────────────────────┐
  │  Linear(128→64) → ReLU → Linear(64→32) → ReLU → Linear(32→C)   ⇒ logits          │
  └─────────────────────────────────────────────────────────────────────────────────┘
```

### A3.1.1 Encoder as a learned filterbank

The encoder is a stack of convolutional blocks, each block

$$
\text{Conv1d}(k{=}5,\ \text{pad}{=}\text{same}) \ \to\ \text{ReLU}\ \to\ \text{MaxPool1d}(k{=}2)\ \to\ \text{BatchNorm1d},
$$

with an optional dropout on every block except the last (the "head" block).
Because padding is `same` and stride is 1, each convolution preserves the time
axis, and the `MaxPool1d(k=2, s=2)` halves it: after $L$ blocks the temporal
length is $T / 2^{L}$. A representative configuration uses channel widths
`[2048, 1024, 512]`, kernel size 5, ReLU activation. Conceptually this is a
*learned filterbank*: rather than fix the front-end filters (as a classical
mel or gammatone bank would), the convolution kernels are trained jointly with
the classifier, so the network is free to discover the band structure that best
separates the classes. `BatchNorm1d` after each pool stabilises the activation
statistics across the wide channel counts.

### A3.1.2 Recurrent core: the LSTM and its gates

The pooled feature sequence is read by a stack of LSTMs (`batch_first=True`),
widths `[256, 128]`. Only the final hidden state $h_T$ of the last layer is
passed forward, so the recurrent core compresses the whole cardiac-cycle
sequence into one vector. This is where the *systole/diastole* temporal
structure is modelled: an LSTM carries a gated memory cell across time steps,
which is what lets it represent the regular S1 → systole → S2 → diastole
rhythm and, by contrast, the irregular timing of an extrasystole.

For an input $x_t$ at time step $t$, previous hidden state $h_{t-1}$, and
previous cell state $c_{t-1}$, the LSTM computes four gates and two state
updates:

$$
\begin{aligned}
i_t &= \sigma\!\left(W_i x_t + U_i h_{t-1} + b_i\right) & &\text{(input gate)}\\
f_t &= \sigma\!\left(W_f x_t + U_f h_{t-1} + b_f\right) & &\text{(forget gate)}\\
g_t &= \tanh\!\left(W_g x_t + U_g h_{t-1} + b_g\right) & &\text{(candidate cell)}\\
o_t &= \sigma\!\left(W_o x_t + U_o h_{t-1} + b_o\right) & &\text{(output gate)}
\end{aligned}
$$

$$
c_t = f_t \odot c_{t-1} + i_t \odot g_t,
\qquad
h_t = o_t \odot \tanh(c_t),
$$

where $\sigma$ is the logistic sigmoid, $\odot$ is the Hadamard (elementwise)
product, and $\{W_\bullet, U_\bullet, b_\bullet\}$ are the learned input,
recurrent, and bias parameters of each gate. The forget gate $f_t$ controls how
much of the previous cardiac-cycle context survives, the input gate $i_t$ and
candidate $g_t$ write new information, and the output gate $o_t$ exposes the
part of the cell relevant to the current decision. The additive cell recurrence
$c_t = f_t \odot c_{t-1} + i_t \odot g_t$ is what protects the gradient across
the long 10-second sequence, avoiding the vanishing-gradient failure of a plain
RNN.

### A3.1.3 Decoder

The final hidden vector is decoded by a small MLP: `Linear(128→64) → ReLU →
Linear(64→32) → ReLU → Linear(32→C)`, emitting raw logits for $C$ classes (no
terminal softmax; the cross-entropy loss applies it internally). Representative
depths are decoder `[64, 32, C]`. The `forward` returns a squeezed logit tensor
of shape $(C,)$ per example.

---

## A3.2 CardioNet V2 — multimodal residual-recurrent experiment (`CardioNetV2`)

V2 keeps the recurrent front end but replaces the shallow LSTM stack with a
*bidirectional* encoder feeding a *pre-activation residual network*, and adds a
second, tabular branch fused by a dense mixer. It is the most elaborate member
of the family, and it carries one caveat that must be stated before anything
else.

> **CRITICAL — V2 is an architectural experiment, not a validated multimodal
> diagnostic model.** The tabular branch was wired to an external
> population-survey source (`CDC_survey_2020.csv`, a CDC/BRFSS-style
> questionnaire, 50 label-encoded features). **The survey respondents are not
> the same people as the subjects of the PCG recordings** (which come from
> PhysioNet/CinC 2016 and PASCAL). The audio–table pairs are therefore *not*
> per-subject aligned: there is no individual whose heart sound and whose
> questionnaire both appear in a training pair. Any multimodal fusion trained
> this way learns, at best, a population prior spuriously attached to unrelated
> audio. V2 is consequently reported as an *experiment in multimodal fusion
> architecture*, demonstrating the residual-recurrent audio path and the mixer
> mechanism; **we make no claim of a working multimodal diagnostic result from
> it.** The project's headline metrics (README §10) come from audio-only runs
> (the best run had `extra_filepath = None`).

```
  audio: MFCC (C, F)                              tabular: survey vector (D)
      │                                                  │
      ▼                                                  │
  Bi-LSTM(input=F, hidden=H, num_layers=2, batch_first) │
      │  output (C, 2H)                                  │
      ▼                                                  │
  pre-activation ResNet-1D (stem → 4 residual groups → GAP)   │
      │  (out_features = 512)                            │
      ▼                                                  ▼
  ┌──────────────────────── DenseMixer ─────────────────────────┐
  │   audio_fc(512→…)          tabular_fc(D→…)                   │
  │            └── concat ──┘                                    │
  │                 mixer FC stack → Linear(→C)  ⇒ logits        │
  └─────────────────────────────────────────────────────────────┘
```

### A3.2.1 Bidirectional LSTM audio encoder

The audio branch is a bidirectional LSTM (README §8.2), `batch_first`, default
`num_layers = 2`. Its hidden size is auto-sized from the feature dimension
$n_{\text{mfcc}} = F$ (the number of MFCCs) as implemented in
`__get_hidden_size`:

$$
H = \left\lfloor \tfrac{2}{3}\, n_{\text{mfcc}} \right\rfloor + n_{\text{mfcc}}.
$$

Because the LSTM is bidirectional it emits $2H$ features per time step
(forward and backward states concatenated), so the tensor handed to the
residual network has channel dimension $2H$ (`__get_resnet_input_shape`
returns `(C, 2H)`). The gate equations are exactly those of §A3.1.2, applied
once in each time direction.

### A3.2.2 Pre-activation residual network (1-D)

The recurrent output feeds a 1-D residual network built by
[`utils/resnet.py`](../cardionix/models/utils/resnet.py). It follows the
*identity-mapping* (pre-activation) formulation of He et al. 2016
[[He2016]](REFERENCES.md), in which each unit orders its operations
**BatchNorm → ReLU → Conv** (the normalisation and activation come *before* the
convolution, not after), so that the shortcut path is a clean identity and the
signal (and gradient) can propagate through the whole stack unmodified.

A pre-activation residual unit computes

$$
y = x + \mathcal{F}(x;\, W),
\qquad
\mathcal{F}(x;\,W) = W_2 * \phi\big(\text{BN}(W_1 * \phi(\text{BN}(x)))\big),
$$

where $\phi$ is ReLU, $*$ is 1-D convolution (kernel 3, padding 1), and
$\mathcal{F}$ is the two-convolution residual branch. When a group changes
resolution, the first unit downsamples (stride-2 first convolution) and the
shortcut becomes a $1\times1$ **projection** (`BatchNorm1d → Conv1d(k=1,
stride=2)`) so the identity term matches the new shape; otherwise the shortcut
is the bare identity. The additive form $y = x + \mathcal{F}(x)$ is the whole
point: $\partial y / \partial x = 1 + \partial \mathcal{F}/\partial x$, so the
gradient always has a direct path and deep stacks stay trainable.

The backbone follows the ResNet-18 layout, four residual groups with channel
widths and unit counts

$$
\{64{:}2,\ 128{:}2,\ 256{:}2,\ 512{:}2\},
$$

preceded by a stem (`Conv1d(k=7, stride=2) → MaxPool1d(k=3, stride=2)`) and
followed by **global average pooling** over the time axis (`GlobalAvgPool1d`,
i.e. $\text{mean}_t$), which collapses the $(512, T')$ feature map to a
512-dimensional vector regardless of sequence length. The first group runs at
full resolution; each later group downsamples once. A `resnet34` preset
$\{64{:}3, 128{:}4, 256{:}6, 512{:}3\}$ is also available.

### A3.2.3 DenseMixer fusion and classifier

The `DenseMixer` ([`utils/mixer.py`](../cardionix/models/utils/mixer.py))
implements late fusion. It builds two independent dense stacks, one per
modality (`audio_fc`, `tabular_fc`), projects each modality's features, then
**concatenates** the two projected vectors (`Concat1d`, along the feature axis)
and passes the result through a mixer FC stack ending in `Linear(→C)`. Formally,
with audio embedding $a$ and tabular embedding $t$,

$$
z = \big[\, f_{\text{audio}}(a) \,\Vert\, f_{\text{tab}}(t) \,\big],
\qquad
\hat{y} = \text{MixerFC}(z),
$$

where $\Vert$ is concatenation. If no tabular tensor is supplied, `Concat1d`
returns the audio branch alone, so the same class degrades cleanly to an
audio-only classifier (which is the mode of the reported best run). An optional
terminal `Softmax` is available (`from_logites=False`); by default the mixer
returns logits.

---

## A3.3 CardioNet V3 — spectral-image transformer (design / prototype stage)

> **Status.** V3 is at the design and prototyping stage. The model source
> [`cardionix/models/cardionetV3.py`](../cardionix/models/cardionetV3.py) is a
> stub, the design note is [`CardionetV3.md`](CardionetV3.md), and prototyping
> lives in `notebooks/Development-CardionetV3-Model.ipynb`. **No V3 result is
> reported**; this section documents intent, not measured performance.

V3 abandons the 1-D view of V1/V2. Instead of feeding a sequence to a recurrent
model, it renders each recording into three complementary time-frequency
representations, stacks them as a three-channel "image" analogous to RGB, and
processes that image with a CNN backbone followed by a Transformer encoder.

```
  audio (10 s)
      │
      ├── STFT magnitude ───────┐
      ├── mel-spectrogram ──────┼──► stack + resize ──►  X : [3, 128, 128]
      └── CWT (Morlet) ─────────┘
                                              │
                                              ▼
                               ResNet-18 backbone  → feature grid [512, H', W']
                                              │
                                              ▼
                          flatten grid → token sequence
                                              │
                                              ▼
                          Transformer encoder (self-attention)
                                              │
                                              ▼
                              classification head  ⇒ logits (C)
```

### A3.3.1 The three channels and what each contributes

Each channel is a different time-frequency decomposition of the same signal,
chosen so that their strengths are complementary:

| Channel | Transform | What it contributes |
|---|---|---|
| 1 | **STFT** magnitude | Fixed-window Fourier view. Resolves the **stationary harmonic structure** of the periodic cardiac rhythm (the frequency content of S1/S2 and of a sustained murmur). |
| 2 | **Mel-spectrogram** | STFT projected onto the mel scale $m(f) = 2595\log_{10}(1 + f/700)$. Applies **perceptual weighting**, compressing high frequencies and emphasising the low band where diagnostic energy concentrates. |
| 3 | **CWT (Morlet)** | Multi-resolution wavelet view. Gives **high temporal resolution on transients** (murmur onsets, clicks, contact artifacts) that a fixed STFT window smears; the Morlet wavelet, a Gaussian-modulated sinusoid, matches the pseudo-sinusoidal shape of heart sound. |

The STFT gives stationary harmonics, the mel channel weights them by perceptual
salience, and the wavelet channel resolves the sharp non-stationary events. A
2-D CNN over the stacked tensor can then exploit correlations *across* the three
views at the same time-frequency location.

### A3.3.2 The wavelet channel: analysis scales

The continuous wavelet transform trades a fixed window for a scalable one:
low-frequency content is analysed with wide wavelets (good frequency
resolution) and high-frequency content with narrow ones (good time resolution).
Scale is inverse to frequency. For a Morlet wavelet of centre frequency $f_c$,
sampling rate $f_s$, and a target frequency grid $f$ (linear from ~20 Hz to
Nyquist), the analysis scale is

$$
a = \frac{f_c \, f_s}{f}.
$$

Higher scale $\Rightarrow$ lower analysed frequency. Evaluating the CWT across
this scale set and resizing yields the $128 \times 128$ wavelet channel.

### A3.3.3 Backbone, tokenisation, and Transformer encoder

The stacked tensor $X \in \mathbb{R}^{3 \times 128 \times 128}$ passes through a
**ResNet-18** backbone (here 2-D, image-style) which extracts local spatial
features and returns a feature grid of shape $[512, H', W']$. That grid is
**flattened into a token sequence** (one token per spatial location, 512-dim),
and a **Transformer encoder** [[Vaswani2017]](REFERENCES.md) models long-range
dependencies between spectral regions that a purely convolutional receptive
field would not connect. Each encoder layer applies multi-head self-attention

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_k}}\right) V,
$$

with $Q, K, V$ linear projections of the token sequence and $d_k$ the key
dimension, followed by a position-wise feed-forward block, residual connections,
and layer normalisation. A final classification head maps the pooled token
representation to the $C$ class logits.

The division of labour is deliberate: the **CNN captures local** time-frequency
texture, the **Transformer captures global** relationships between distant
spectral regions across the cardiac cycle.

### A3.3.4 Provenance: NAS/literature survey

The V3 design was preceded by a neural-architecture-search and literature
survey, preserved in [`NAS-For-V3.md`](NAS-For-V3.md), which reviews 2-D CNNs on
(mel-)spectrograms, CNN+RNN hybrids, combined-TFR one-shot CNNs (STFT + mel +
wavelet stacked, the direct antecedent of the V3 three-channel idea),
attention-CNNs, mixture-of-experts ensembles, and the end-to-end learnable-
filterbank SpectNet. The three-channel spectral-image-plus-transformer design is
the synthesis that survey pointed to.

---

## A3.4 Experimental utilities around the family

Beyond the three published architectures, the model package carries several
experimental building blocks used in exploration and not (necessarily) in the
best run:

- **ONNX export** (`notebooks/cardionetv2_onnx.ipynb`): a working export path for
  V2, intended for on-device / offline inference on the phone that made the
  recording (README §13, item 5).
- **TabNet** (attentive tabular learning): an experimental alternative to the
  dense tabular branch of V2.
- **GhostBatchNorm** ([`utils/gbn.py`](../cardionix/models/utils/gbn.py)):
  batch normalisation over virtual sub-batches, which stabilises training under
  large batch sizes.
- **Sparsemax** ([`utils/sparsemax.py`](../cardionix/models/utils/sparsemax.py)):
  a sparse, normalised alternative to softmax that can assign exactly zero
  probability to some classes, useful for interpretable attention and output
  distributions.

These are infrastructure for the next round of experiments (V3 and the
data-centric work of README §11), not components of the reported best model.

---

### References (see [`REFERENCES.md`](REFERENCES.md))

- **[He2016]** He K., Zhang X., Ren S., Sun J. *Identity Mappings in Deep
  Residual Networks.* ECCV, 2016. arXiv:1603.05027 — the pre-activation
  (BatchNorm → ReLU → Conv) residual unit of §A3.2.2.
- **[Vaswani2017]** Vaswani A. et al. *Attention Is All You Need.* NeurIPS,
  2017. arXiv:1706.03762 — the Transformer encoder of §A3.3.3.
