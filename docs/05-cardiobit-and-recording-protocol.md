# Appendix 5 — cardiobit, cardiolab-studio, and the recording protocol

> Companion appendix to [§11 of the README](../README.md#11-the-turn-to-a-data-centric-method-cardiobit-and-the-recording-protocol). This is the "we hit a wall and built tooling to get past it" part of the story, told at full length. It documents the two spin-off tools, [`cardiobit`](https://github.com/Cardionix/cardiobit) (v0.1.1) and [`cardiolab-studio`](https://github.com/Cardionix/cardiolab-studio), the proxy signal-quality metric that let us rank preprocessing configurations, and the acquisition experiment that produced the recording protocol.

---

## 5.1 Why a separate library

The architectural search of [§10](../README.md#10-experiments-and-results) reached a plateau. Once the class taxonomy was fixed at three clinically meaningful labels, further changes to network depth, recurrent width, and hyperparameters stopped producing gains: the learning curves flattened regardless of capacity. The standard reading applies. When the model class is already expressive enough to fit the signal, additional capacity buys nothing, and the binding constraint moves to the volume of data and the separability of the representation.

The representation was the limiting factor. The natural but wrong response would have been to bolt one more preprocessing function onto the training pipeline, tune it against the validation macro-F1, and move on. That is wrong for two reasons. First, it conflates two feedback loops with very different time constants: a training run costs minutes to hours, whereas a preprocessing hypothesis ("does a tighter passband improve S1/S2 contrast?") should be answerable in seconds. Second, coupling signal-processing experiments to the training loop makes them non-reusable: every insight would be locked inside `cardionix-pipeline` and unavailable to the acquisition and quality-assessment work that was becoming the centre of gravity of the project.

The signal-processing work had also outgrown its container. Early notes on a universal smartphone-PCG preprocessor, preserved in [`docs/R.MD`](R.MD) and [`docs/R4o.MD`](R4o.MD), started as inline preprocessing sketches and grew into filter-design decisions, noise-floor estimators, quality metrics, and biometric extractors. That is a library, not a pipeline stage. So the signal-processing work was factored out into a standalone, reusable package, [`cardiobit`](https://github.com/Cardionix/cardiobit) (v0.1.1, roughly 1,177 lines of code; dependencies `scipy 1.15.2`, `numpy 2.2.4`, `librosa 0.11.0`, `soundfile 0.13.1`, `noisereduce`, `PyWavelets`).

The library serves a dual goal, and both halves are about moving load off the model:

- **(a) Feature extraction that increases class separability by reducing representation complexity.** If the front end can present a cleaner, band-limited, denoised signal, the classifier has less nuisance variance to learn around, and the same capacity buys more discrimination. This moves work from the model to the front end, where it is cheaper and more inspectable.
- **(b) Domain-specific augmentation that widens corpus variance without distorting diagnostic content.** Augmentation must broaden the distribution the model sees (device, noise, contact conditions) while leaving the diagnostic evidence (S1/S2 timing and relative amplitude, murmur texture) intact. A generic audio augmentation that, for example, aggressively pitch-shifts or low-passes can destroy exactly the evidence being classified.

The remainder of this section documents the DSP toolkit that implements goal (a); the quality and biometric instrumentation that makes its effects measurable; the proxy metric that turns "did it help?" into a rankable number; and the interactive workbench that closes the loop.

A design principle runs through the whole toolkit: every routine that can silently destroy diagnostic content is instrumented to warn when its parameters leave the safe region, and every quality number it reports is tagged with a trust level (§5.3.3). The library is built for a screening context, where a plausible-looking but wrong output is more dangerous than an obviously missing one, so the defaults are conservative and the aggressive settings are gated behind explicit warnings rather than left as quiet footguns.

---

## 5.2 The cardiobit DSP toolkit

All routines operate on a mono, floating-point signal $y[n]$ sampled at rate $f_s$. The physical target is the cardiac acoustic band, roughly $20$ to $200$ Hz, established by the propagation model of [§3](../README.md#3-problem-framing-why-smartphone-pcg-is-not-ordinary-audio-ml):

$$
x(t) = \big(s * h_{\text{tissue}}\big)(t) + n_{\text{env}}(t) + n_{\text{contact}}(t),
$$

where $s(t)$ is the valve-level acoustic source, $h_{\text{tissue}}(t)$ the tissue propagation impulse response, and $n_{\text{env}}, n_{\text{contact}}$ the environmental and contact noise terms. Every routine below is a hypothesis about how to invert or suppress one of these terms without damaging the diagnostic content of $s$.

### 5.2.1 Loading and resampling (`io.py:85`)

`load` wraps `librosa.load`: mono downmix, resampling to a default $f_s = 2000$ Hz (optimal range $2000$ to $4000$ Hz, matched to the $20$ to $200$ Hz diagnostic band and to the PhysioNet/CinC 2016 convention), with optional `offset`/`duration` windowing. The target rate is Nyquist-checked: since the highest band of interest is $200$ Hz, any $f_s \geq 400$ Hz satisfies the sampling theorem for the signal band, and $2000$ Hz leaves generous headroom for the anti-alias transition while keeping tensors small. Requesting a rate that would place the $200$ Hz upper edge above the new Nyquist frequency raises a warning.

### 5.2.2 Bandpass Butterworth filter (`filter.py:45`)

A zero-phase Butterworth IIR bandpass over the cardiac band. Defaults: `lowcut` $= 20$ Hz, `highcut` $= 200$ Hz, `order` $n = 4$. The analogue prototype has the maximally flat magnitude response

$$
|H(j\omega)|^2 = \frac{1}{1 + \left(\dfrac{\omega}{\omega_c}\right)^{2n}},
$$

so the passband is as flat as possible (no ripple), at the cost of a gentler roll-off than a Chebyshev or elliptic design of equal order. Maximal passband flatness is the right trade for PCG: ripple in the passband would colour the relative S1/S2 amplitudes, which one of the biometric metrics (§5.3) reads directly.

Two implementation choices matter diagnostically:

- **Zero-phase filtering (`filtfilt`).** The signal is filtered forward and then backward, which squares the magnitude response (so the effective order is $2n$) and, crucially, cancels the phase response to exactly zero. This is essential because S1/S2 timing and the systole/diastole interval carry rhythm information (a naive single-pass IIR filter introduces frequency-dependent group delay that smears those transients and shifts the very peak positions that the beat detector and the S1/S2 ratio of §5.3 depend on). An `sosfilt` (second-order-sections) path is also available for the streaming case where the whole signal is not in memory, at the cost of reintroducing phase delay.
- **Nyquist and band-sanity checks.** The routine validates that `highcut` is below the Nyquist frequency $f_s/2$, and warns when the requested band falls outside a plausible $10$ to $400$ Hz window, since a passband far from the cardiac band almost always signals a misconfiguration rather than an intentional choice.

The order-4 default is a compromise. Higher orders sharpen the transition band and reject more out-of-band noise, but they push the IIR poles closer to the unit circle, which worsens numerical conditioning and lengthens the impulse response (more transient ringing). Order 4, doubled to an effective 8 by `filtfilt`, gives adequate stopband rejection at the $20$ and $200$ Hz edges while keeping the ringing short relative to the S1/S2 spacing.

### 5.2.3 Wiener adaptive filter (`filter.py:16`)

`wiener_filter` wraps `scipy.signal.wiener`, an adaptive, locally-stationary Wiener filter. Over a sliding window of size `mysize` (default range $5$ to $15$ samples), it estimates the local mean $\mu$ and variance $\sigma^2$ and attenuates each sample toward the local mean in proportion to the estimated noise power $\nu^2$:

$$
\hat{y}[n] = \mu + \frac{\max(\sigma^2 - \nu^2,\, 0)}{\sigma^2}\,\big(y[n] - \mu\big),
$$

with $\nu^2$ taken as the mean of the local variances. Where the local variance is high (a true heart sound), the filter passes the sample nearly unchanged; where it is low (quiet noise floor), it pulls toward the local mean. Small windows preserve transient sharpness at the cost of less smoothing; larger windows smooth more but can round the S1/S2 onsets.

### 5.2.4 Non-stationary spectral gating (`denoising.py:16`)

`spectral_gating` wraps `noisereduce` in its non-stationary mode (`stationary=False`), which estimates a time-varying noise floor and gates each time-frequency bin below its local threshold. This is the right model for smartphone captures, where $n_{\text{env}}$ and $n_{\text{contact}}$ are emphatically not stationary (speech onsets, cloth friction bursts, motion). Parameters:

| Parameter | Default | Role |
|---|---|---|
| `prop_decrease` | $0.5$ | Fraction of the estimated noise removed (0 = none, 1 = maximal) |
| `time_constant_s` | $1.0$ s | Time window over which the non-stationary noise estimate adapts |
| `freq_mask_smooth_hz` | $100$ Hz | Frequency smoothing of the gate mask, to avoid musical noise |
| `n_fft` | $2048$ | STFT window length |
| `win_length` | $1024$ | Analysis window |
| `hop_length` | $256$ | Hop between frames |

`prop_decrease = 0.5` is deliberately conservative. The routine carries rich domain-warnings: setting `prop_decrease > 0.8` risks suppressing S1/S2 and, worse, low-amplitude murmurs, which are precisely the diagnostic content the classifier depends on. In a screening context, over-denoising is a diagnostic error, not merely an aesthetic one, so the aggressive settings are gated behind an explicit warning rather than silently permitted.

### 5.2.5 Wavelet denoise (`denoising.py:158`)

`wavelet_denoise` performs discrete wavelet transform (DWT) shrinkage. Defaults: wavelet `db6` (Daubechies-6, whose support and vanishing moments suit the sharp-onset, oscillatory morphology of heart sounds), decomposition `level` $= 3$, soft thresholding. The threshold is the Donoho–Johnstone universal threshold:

$$
u = \sigma \sqrt{2 \ln N},
$$

where $N$ is the signal length and $\sigma$ is a robust estimate of the noise standard deviation from the finest-scale detail coefficients $c_D$:

$$
\sigma = \frac{\operatorname{median}\big(|c_D|\big)}{0.6745}.
$$

The constant $0.6745$ is the $0.75$ quantile of the standard normal, so $\operatorname{median}(|c_D|)/0.6745$ is the median-absolute-deviation estimator of $\sigma$, which is robust to the sparse large coefficients that correspond to real signal rather than noise. Soft thresholding

$$
\eta_u(c) = \operatorname{sign}(c)\,\max\big(|c| - u,\; 0\big)
$$

shrinks coefficients continuously toward zero (rather than the discontinuous hard rule), which avoids the ringing artifacts that hard thresholding introduces near transients. Soft thresholding does bias the surviving coefficients slightly toward zero, which mildly attenuates true signal, but for PCG the smoothness near S1/S2 onsets is worth that bias.

The three denoising paths (§5.2.3 Wiener, §5.2.4 spectral gating, §5.2.5 wavelet) are interchangeable and address different noise structures, which is why the library offers all three rather than picking one. Wiener is a time-domain, locally-adaptive filter that excels at smoothing stationary broadband noise but has no frequency selectivity. Spectral gating is a time-frequency method that shines against non-stationary interference (speech, friction bursts) because it can gate individual bins as they appear and vanish. Wavelet shrinkage sits between them: it is transient-aware by construction, so it denoises without smearing the sharp S1/S2 onsets, but its single global threshold is less adaptive to time-varying noise than spectral gating. In practice the studio (§5.5) lets these be chained and compared directly, which is the point of shipping them as separate, composable stages rather than one fused denoiser.

### 5.2.6 Normalization (`preprocess.py:13`)

`normalize` scales to the unit interval $[-1, 1]$:

$$
y_{\text{norm}}[n] = \frac{y[n]}{\max_m |y[m]|}.
$$

This removes device- and gain-dependent amplitude scale, which is nuisance variance for classification, and it is a precondition for the clipping metric (§5.3), which is defined against the normalized signal.

### 5.2.7 Visualization (`visual/visual.py`)

`plot(method=...)` renders `waveform`, `spectrogram` (STFT, `n_fft` $= 2048$, `hop` $= 512$, dB-scaled), or `melspectrogram` (`n_mels` $= 128$, dB-scaled). These are the three views `cardiolab-studio` composes into its before/after panels (§5.5).

---

## 5.3 Signal-quality and biometric metrics

Preprocessing without measurement is guesswork. `cardiobit` ships two metric families: **signal-quality** metrics (`metrics/device.py`), which score the recording as a recording, and **biometric** metrics (`metrics/biomonitor.py`), which attempt to read physiology out of it. Both families feed the interactive workbench and the proxy metric of §5.4.

### 5.3.1 Signal-quality metrics (`metrics/device.py`)

**SNR** (`device.py:20`). The signal is median-filtered (kernel $k = 5$) to suppress impulsive spikes, then the noise power $P_n$ is estimated from the $10$th percentile of $|y|$ (a floor that is robust to the loud heart-sound bursts), and the signal power $P_s$ from the full signal:

$$
\text{SNR} = 10 \log_{10}\!\left(\frac{P_s}{P_n}\right).
$$

Interpretation: $> 20$ dB clean, $10$ to $20$ dB moderate, $< 10$ dB poor.

**Dynamic range** (`device.py:88`). Over samples with $|y| > 10^{-6}$ (excluding true silence),

$$
\text{DR} = 20 \log_{10}\!\left(\frac{\max|y|}{\min|y|}\right),
$$

with $> 60$ dB indicating good preservation of both faint and loud events.

**Clipping %** (`device.py:155`). The fraction of samples of the normalized signal that sit at or beyond the rails:

$$
\text{Clip} = \frac{\big|\{\, n : |y_{\text{norm}}[n]| \geq 0.99 \,\}\big|}{N} \times 100\%,
$$

with $> 5\%$ flagged as unreliable (hard nonlinear distortion irreversibly corrupts the waveform).

**Spectral purity** (`device.py:216`). From the rFFT, the fraction of total energy that falls inside the cardiac band:

$$
\text{Purity} = \frac{E_{[20,\,200]\text{ Hz}}}{E_{\text{total}}}.
$$

This is an in-band energy-concentration measure and serves as a lightweight PEAQ-style proxy: a recording whose energy is dominated by out-of-band content is dominated by noise.

**Spectral entropy** (`device.py:272`). The Shannon entropy of the normalized power spectrogram (`nperseg` $= 256$), averaged over frames:

$$
H = -\sum_{f} P(f) \log_2 P(f), \qquad \sum_f P(f) = 1,
$$

where $P(f)$ is the normalized power at frequency bin $f$. A quasi-periodic clean heart sound concentrates its power in a few bins (low entropy); broadband noise spreads it (high entropy). Interpretation: $< 2$ clean, $> 4$ noisy.

### 5.3.2 Biometric metrics (`metrics/biomonitor.py`)

**Heart rate** (`biomonitor.py:16`). The beat detector is a Hilbert-envelope cascade: bandpass to the cardiac band, take the analytic-signal envelope $e[n] = |y[n] + j\,\mathcal{H}\{y\}[n]|$ via the Hilbert transform, then `scipy.signal.find_peaks` with `distance` $= 0.3$ s (a physiological refractory floor, capping detectable rate at $200$ bpm) and `prominence` $= 0.1 \cdot \max e$. With at least `min_peaks` $= 3$ detected beats at inter-peak intervals $\Delta T_i$,

$$
\text{HR} = \frac{60}{\overline{\Delta T}} \quad \text{[bpm]}.
$$

Normal adult resting range is $60$ to $100$ bpm.

**HRV (SDNN)** (`biomonitor.py:86`). The standard deviation of the RR intervals (in milliseconds), after filtering intervals to the plausible physiological window $300$ to $2000$ ms (excluding missed and spurious beats):

$$
\text{SDNN} = \operatorname{std}\big(\{\text{RR}_i\}\big).
$$

**S1/S2 amplitude ratio** (`biomonitor.py:152`). From the envelope peaks (after a $1$st–$99$th percentile outlier trim, `min_peaks` $= 4$), peaks are assigned to the alternating S1 (even index) and S2 (odd index) sequence, and

$$
r_{\text{S1/S2}} = \frac{\overline{A_{\text{S1}}}}{\overline{A_{\text{S2}}}}.
$$

A ratio near $1$ is typical; departures can be physiological or pathological, but the metric is only as good as the S1/S2 assignment (see the trust level below).

### 5.3.3 Per-metric trust levels

Every metric carries an explicit trust level reflecting its sensitivity to noise and to upstream assumptions. This is unusual and deliberate. A quality tool that hides its own uncertainty is worse than useless in a screening context: it invites a clinician or an app to act on a number that the underlying signal does not support. The point of surfacing per-metric uncertainty is that a single recording rarely fails everywhere at once; a recording can have honest, high-trust dynamic range and clipping while its S1/S2 ratio is meaningless because the beats were never cleanly separated. Reporting one aggregate "quality score" would hide exactly the distinction the downstream decision needs.

| Metric | Trust | Why |
|---|---|---|
| Dynamic range | **High** | Simple, robust; a direct amplitude statistic |
| Clipping % | **High** | Trivially verifiable; a hard, unambiguous count |
| SNR | **Conditional** | Depends on the percentile noise-floor estimate being valid |
| Heart rate | **High if clean** | Reliable on a clean signal; degrades sharply once the envelope peaks are corrupted |
| HRV (SDNN) | **Medium** | Noise-sensitive; a single missed or spurious beat shifts SDNN |
| S1/S2 ratio | **Low** | Requires correct S1/S2 separation, which is fragile |
| Spectral entropy | **Medium** | Depends on the `nperseg` segmentation choice |
| Spectral purity | **Medium** | Depends on the fixed $20$ to $200$ Hz band assumption |

The two high-trust metrics (dynamic range, clipping) are pure amplitude statistics with no physiological assumptions, so they hold even on a noisy recording. SNR is conditional because its value is only as good as the noise-floor estimate: the $10$th-percentile heuristic is defensible but not guaranteed. Heart rate is high-trust exactly when the recording is clean and degrades precisely when it is not, which makes it a natural building block for the proxy metric below. The biometric ratios sit lowest because they inherit every error of the beat detector and then add the S1/S2 assignment on top.

---

## 5.4 The proxy signal-quality metric

There is no direct ground truth for the question that actually drove the DSP work: *did this processing step improve the signal?* There is no reference "clean" version of a smartphone recording to compare against, and human judgement of spectrograms is slow, subjective, and does not scale to ranking dozens of preprocessing configurations.

We introduced an indirect, operational measure. The observation is simple: on a cleaner signal, an automatic beat detector agrees more closely with a human's manual count of heartbeats. The Hilbert-envelope + `find_peaks` cascade of §5.3.2 is that detector; a person listening to (or watching the waveform of) the recording provides the reference count $N_{\text{manual}}$. The proxy quality is the negative absolute discrepancy:

$$
Q = -\,\big|\, N_{\text{manual}} - N_{\text{algo}} \,\big|,
$$

so $Q = 0$ is perfect agreement and $Q$ becomes more negative as the detector miscounts. A preprocessing pipeline that reduces the count error raises $Q$, and is, by this proxy, improving the signal. Concretely, this let us rank preprocessing configurations against each other on a fixed set of recordings, replacing "argue about spectrograms by eye" with a scalar to sort by.

Two clarifications on where this lives. The metric is **conceptual/operational**, not a standalone exported function in `cardiobit` v0.1.1: the algorithmic count $N_{\text{algo}}$ is produced by the existing Hilbert-envelope beat detector, and the comparison to the manual count is the evaluation procedure applied on top of it. It is a way of using the library, not a new primitive inside it.

**Limitations.** The proxy is deliberately crude and should be read as an ordinal ranking aid, not a calibrated quality score.

- **Not injective in the right direction.** A pipeline can improve $Q$ by making the detector count correctly while simultaneously distorting diagnostic content the count does not measure. $Q$ scores *beat detectability*, which correlates with cleanliness but is not identical to *diagnostic fidelity*. An over-aggressive denoiser that flattens the signal into clean, evenly spaced blobs could score a perfect $Q$ while having destroyed the murmur.
- **Coarse resolution.** $N$ is a small integer over a short recording, so $Q$ takes only a handful of discrete values and cannot distinguish two configurations that both count correctly. It separates good from bad, not good from slightly better.
- **Reference cost and subjectivity.** $N_{\text{manual}}$ requires a human per recording and is itself error-prone on genuinely ambiguous captures (which are exactly the hard cases). The proxy is only as trustworthy as the manual count.
- **No pathology awareness.** Arrhythmias and dropped beats can make the "true" count itself ambiguous, so $Q$ is most reliable on the regular-rhythm recordings where it is least needed.

Despite this, $Q$ did its job: it converted an unmeasurable question into a rankable one, which is what let the preprocessing search proceed at all. The right way to read it is as a *filter*, not a *ranker*: a configuration with a badly negative $Q$ can be rejected with confidence (the detector cannot even find the beats, so the recording is unusable or the pipeline mangled it), while among configurations that all score near zero, the tie is broken by the higher-trust quality metrics of §5.3 and, ultimately, by listening. Used that way, $Q$ complements the trust-tagged metrics rather than replacing them: it catches gross failures cheaply and defers the fine distinctions to measures whose uncertainty is better understood.

---

## 5.5 cardiolab-studio: making the comparison interactive

`Q` gives a number; [`cardiolab-studio`](https://github.com/Cardionix/cardiolab-studio) gives the loop that produces the number quickly. It is a Streamlit workbench (Streamlit 1.44.1; depends on `librosa 0.11`, `scipy 1.15.2`, `numpy 2.2.4`, `cardiobit 0.1.1`) whose entire purpose is to turn "test a preprocessing hypothesis" from a code edit into a slider drag. Dark theme, green accent (`#138312`).

**Workflow.** The user uploads a WAV or MP3 recording. The app renders two columns, **Before** (original) and **After** (processed), and applies a user-configurable processing chain:

$$
\texttt{load} \;\to\; \texttt{normalize} \;\to\; \texttt{bandpass} \;\to\; \texttt{wiener} \;\to\; \texttt{wavelet} \;\to\; \texttt{custom}.
$$

Each stage is a sidebar expander with its parameters exposed live:

- **load**: sample rate ($1$ k to $22.05$ k Hz), offset, duration;
- **bandpass**: `lowcut`, `highcut`, `order` ($1$ to $10$);
- **wiener**: window size ($5$ to $33$);
- **wavelet**: type (`db6` / `sym5` / `coif3`), level ($2$ to $5$);
- **custom**: a free-form Python preprocessing step.

**What it shows.** For each signal (before and after), the app renders the three `cardiobit` views (waveform, STFT spectrogram in `magma`/dB, mel-spectrogram in `magma`), plus a native `st.audio` player so the recording can be heard before and after (plots are rendered to in-memory PNG via `BytesIO`). Alongside the plots it surfaces the `cardiobit` quality and biometric metrics with their trust levels, so a configuration's effect is visible in the waveform, audible in the player, and quantified in the metric panel simultaneously.

**Extensibility.** Two panels make the workbench itself hackable rather than fixed. A **Plugins** panel installs and uninstalls libraries dynamically, and a **Manual Code Entry** panel accepts custom Python preprocessing code as the final `custom` stage of the chain (evaluated at runtime). This is what let new denoising ideas be tried in the studio without rebuilding it: the studio is the interactive front end, `cardiobit` is the engine underneath.

A candid note on scope. The current studio version composes plots and audio; the live per-recording metric *table* with trust indicators is described in the "How it works?" panel (which enumerates the nine metrics: SNR, dynamic range, clipping, PEAQ proxy, HR, HRV, S1/S2 ratio, spectral entropy, spectral purity) and is computed by `cardiobit`, but is only partially surfaced as UI in v0.1.1. PEAQ in particular is not implemented; the spectral-purity fraction stands in as its proxy. The trust levels themselves are documented qualitatively (in `cardiobit`'s README and in [`docs/package.md`](package.md)) rather than rendered as confidence bars. We state this rather than overclaim a finished dashboard.

---

## 5.6 Acquisition as an experiment: the recording protocol

The propagation model of [§3](../README.md#3-problem-framing-why-smartphone-pcg-is-not-ordinary-audio-ml) is the reason this section exists. In

$$
x(t) = \big(s * h_{\text{tissue}}\big)(t) + n_{\text{env}}(t) + n_{\text{contact}}(t),
$$

the tissue transfer function $h_{\text{tissue}}$ and both noise terms are fixed at capture time. No amount of downstream DSP can recover diagnostic energy that $h_{\text{tissue}}$ attenuated below the noise floor, and no denoiser can cleanly remove $n_{\text{contact}}$ once it overlaps the cardiac band. Acquisition is therefore not preprocessing hygiene; it is a first-class term in the signal model, and controlling it is a lever on accuracy that sits *upstream* of everything `cardiobit` can do.

So acquisition was treated as an experiment in its own right. We varied, one factor at a time and in combination:

- **pickup point** on the torso (which cardiac projection the phone covers);
- **body posture** (upright, supine, reclined);
- **background noise** level and type (quiet room versus street);
- **device model** (which physically changes the microphone and its coupling, and therefore $h_{\text{tissue}}$ at the contact interface);
- **contact pressure** (how firmly the phone is pressed to the chest, which changes coupling and $n_{\text{contact}}$);
- **protective case** on or off (an added compliant layer in the propagation path).

Each configuration was scored with the `cardiobit` quality metrics of §5.3 and the proxy $Q$ of §5.4, which is exactly what made this an experiment rather than an opinion. The three high-to-conditional-trust quality metrics (dynamic range, clipping, SNR) carry most of the weight here: they are the ones that answer "is this a good recording?" without depending on a physiological model, so they are the honest arbiters of an acquisition choice. The biometric metrics were read only as sanity checks, since a configuration that produces an implausible heart rate is more likely broken than diagnostic.

Some of the factors act through $n_{\text{contact}}$ (posture and contact pressure change how the skin-device interface loads and rubs), some through $h_{\text{tissue}}$ (pickup point and the protective case change the propagation path itself), and some through $n_{\text{env}}$ (background noise). Separating them mattered because they are not equally fixable downstream: environmental noise is partly recoverable by the denoisers of §5.2, but a case-induced attenuation of $h_{\text{tissue}}$ and a poor pickup point are not, since they remove energy before it is ever captured.

**Outcome protocol.** The configuration that maximized signal quality was:

1. **Upright posture.**
2. **Pickup in the intercostal space over the cardiac projection** (placing the microphone where the cardiac acoustic energy couples most directly, minimizing the tissue path).
3. **Protective case removed** (eliminating the compliant layer that attenuates and colours the signal, i.e. degrades $h_{\text{tissue}}$).

**The methodological point generalizes beyond these specific settings.** Under a data-constrained regime, improving acquisition quality is a more effective lever than adding model capacity. When the model class is already expressive enough (§5.1) and data are scarce, the marginal return on a better recording exceeds the marginal return on a deeper network: it raises the ceiling on what any model can extract, whereas capacity only chases a ceiling the signal has already set. Put bluntly, it is cheaper to record better than to learn harder. This is the same data-centric conclusion that drove the PhysioNet corpus expansion of [§5.3 of the README](../README.md#53-the-data-centric-expansion-physionet), applied one step further upstream: to the moment of capture itself.

---

## 5.7 Summary

The plateau of the architectural search was a diagnosis, not a dead end. It said the representation and the data, not the model, were binding. The response was a data-centric line of work with three deliverables: `cardiobit`, a reusable PCG DSP and quality-assessment library that moves load onto an inspectable front end and reports its own per-metric uncertainty; the proxy metric $Q$, which made "did processing help?" rankable without a ground-truth clean signal; and `cardiolab-studio`, which made the whole comparison loop interactive. The acquisition experiment closed the argument: since $h_{\text{tissue}}$ and the noise terms are set at capture time, the cheapest large lever left in a data-constrained regime is to record better in the first place.
