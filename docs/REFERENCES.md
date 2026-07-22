# References

Annotated bibliography for the Cardionix research report. Grouped by role.

## Epidemiology and motivation

- **[WHO]** World Health Organization. *Cardiovascular diseases (CVDs) fact sheet.*
  <https://www.who.int/health-topics/cardiovascular-diseases> and
  <https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)>.
  Source for the 17.9 million deaths/year, ~31% of global mortality, 85% from heart attack and stroke, >75% in low- and middle-income countries, ~1/3 premature (<70 y).

- **[GBD 2019]** Roth G. A., Mensah G. A., Johnson C. O., et al. *Global Burden of Cardiovascular Diseases and Risk Factors, 1990–2019: Update From the GBD 2019 Study.* Journal of the American College of Cardiology, 76(25):2982–3021, 2020. doi:[10.1016/j.jacc.2020.11.010](https://doi.org/10.1016/j.jacc.2020.11.010). Absolute CVD deaths rose from 12.1 M (1990) to 18.6 M (2019); prevalent cases 271 M → 523 M.

- **[GBD 2021]** *Global, Regional, and National Burden of Cardiovascular Disease, 1990–2021: Results From the 2021 Global Burden of Disease Study.* Journal of the American College of Cardiology, 2024. PMC:[PMC11668263](https://pmc.ncbi.nlm.nih.gov/articles/PMC11668263/). CVD deaths 12.33 M (1990) → 19.42 M (2021), +57.5%; age-standardised death rate 358.1 → 235.2 per 100k, −34.3%; incident cases 34.74 M → 66.81 M.

- **[IHME]** Institute for Health Metrics and Evaluation. *Global burden of cardiovascular diseases and risks, 1990–2022.* <https://www.healthdata.org/research-analysis/library/global-burden-cardiovascular-diseases-and-risks-1990-2022>. CVD deaths reach 19.8 M in 2022; age-standardised rate −34.9%.

- World Heart Federation. *World Heart Report 2023: Confronting the World's Number One Killer.* <https://world-heart-federation.org/>.

## Datasets

- **[PASCAL]** Bentley P., Nordehn G., Coimbra M., Mannor S. *The PASCAL Classifying Heart Sounds Challenge (CHSC2011).* 2011. <http://www.peterjbentley.com/heartchallenge/>. 832 audio snippets, five classes (normal, murmur, extrasystole, extra heart sound, artifact); set A via the iStethoscope Pro iPhone app, set B via the DigiScope digital stethoscope. Basis of the cleaned **Dangerous Heartbeat Dataset (DHD)**.
  - Gomes E. F., Bentley P., et al. *Classifying Heart Sounds: Approaches to the PASCAL Challenge.* 2013.

- **[PhysioNet2016]** Liu C., Springer D., Li Q., et al. *An open access database for the evaluation of heart sound algorithms.* Physiological Measurement, 37(12):2181–2213, 2016. doi:[10.1088/0967-3334/37/12/2181](https://doi.org/10.1088/0967-3334/37/12/2181). The largest public heart-sound database, aggregated from nine sources; 2435 recordings from 1297 subjects.

- **[Clifford2016]** Clifford G. D., Liu C., Moody B., et al. *Classification of Normal/Abnormal Heart Sound Recordings: the PhysioNet/Computing in Cardiology Challenge 2016.* Computing in Cardiology, 43:609–612, 2016. <https://physionet.org/content/challenge-2016/>.

## Methods and architectures

- **[Springer2016]** Springer D. B., Tarassenko L., Clifford G. D. *Logistic Regression-HSMM-Based Heart Sound Segmentation.* IEEE Transactions on Biomedical Engineering, 63(4):822–832, 2016. doi:[10.1109/TBME.2015.2475278](https://doi.org/10.1109/TBME.2015.2475278). Reference method for S1/S2 segmentation.

- **[He2016]** He K., Zhang X., Ren S., Sun J. *Identity Mappings in Deep Residual Networks.* ECCV, 2016. arXiv:[1603.05027](https://arxiv.org/abs/1603.05027). Pre-activation residual units used in CardioNet V2/V3.

- **[Vaswani2017]** Vaswani A., Shazeer N., Parmar N., et al. *Attention Is All You Need.* NeurIPS, 2017. arXiv:[1706.03762](https://arxiv.org/abs/1706.03762). Transformer encoder for CardioNet V3.

- **[Donoho1994]** Donoho D. L., Johnstone I. M. *Ideal Spatial Adaptation by Wavelet Shrinkage.* Biometrika, 81(3):425–455, 1994. doi:[10.1093/biomet/81.3.425](https://doi.org/10.1093/biomet/81.3.425). Universal threshold and robust noise estimate used by cardiobit's wavelet denoiser.

- **[CardioXNet]** Baghel N., Dutta M. K., Burget R. *CardioXNet: A Novel Lightweight Deep Learning Framework for Cardiovascular Disease Classification Using Heart Sound Recordings.* arXiv:[2010.01392](https://arxiv.org/abs/2010.01392).

- **[SpectNet]** *SpectNet: End-to-End Audio Signal Classification Using Learnable Spectrograms.* arXiv:[2211.09352](https://arxiv.org/abs/2211.09352).

- McFee B., Raffel C., Liang D., et al. *librosa: Audio and Music Signal Analysis in Python.* Proc. 14th Python in Science Conference (SciPy), 2015. MFCC, STFT, mel-spectrogram, HPSS.

## Supplementary surveys

- *A Deep-Learning Approach to Heart Sound Classification.* Technologies, MDPI, 13(4):147. <https://www.mdpi.com/2227-7080/13/4/147>.
- *Heart sound classification based on improved mel-frequency features.* PMC:[PMC9814508](https://pmc.ncbi.nlm.nih.gov/articles/PMC9814508/).
- Sakib S., et al. *ENACT-Heart: Ensemble-based Assessment Using CNN and Transformer on Heart Sounds.* arXiv:[2502.16914](https://arxiv.org/abs/2502.16914).
- *On the analysis of data augmentation methods for spectral-imaged heart-sound classification.* BMC Medical Informatics and Decision Making, 2022. doi:[10.1186/s12911-022-01942-2](https://doi.org/10.1186/s12911-022-01942-2).

> Notes: dataset counts in this report are taken from the project's own harmonised annotation files, which use a subset/rebalanced view of the public sources above. Epidemiological endpoints are quoted verbatim from the cited studies; intermediate years in the mortality figure are interpolated for display and labelled as such in [`make_figures.py`](make_figures.py).
