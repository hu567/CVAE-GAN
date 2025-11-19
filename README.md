# CVAE-GAN: Spectrum-Compatible Artificial Accelerograms

This repository contains the official implementation of the paper:

> **Xiaohu Hu, Su Chen, Yi Ding, Lei Fu, Xiaojun Li (2026)**  
> *Spectrum-compatible artificial accelerograms via conditional variational autoencoder with generative adversarial networks*  
> Engineering Structures, Vol. 348, 121766  
> ISSN 0141-0296  
> https://doi.org/10.1016/j.engstruct.2025.121766  

If you use this code, **please cite the paper above.**

---

## Abstract

The need for spectrum-compatible ground motions in structural seismic design has driven the development of artificial seismic waveform generation techniques. This study proposes a Conditional Variational Autoencoder with Generative Adversarial Networks (CVAE-GAN) framework to generate artificial accelerograms conditioned on acceleration response spectra, addressing the demand for spectrum-compatible ground motions in earthquake engineering. Utilizing Japan’s K-NET and KiK-net seismic records, preprocessed with PhaseNet for P-wave detection, the model generates diverse accelerograms that preserve the temporal and spectral characteristics of real earthquakes. Validation on a test set (4.5 ≤ Mw ≤ 7.5) demonstrates that over 99% of the generated spectra achieve an R² above 0.8 with an RMSE below 0.1 m/s², confirming its high accuracy and realism. Furthermore, the model exhibits generalizability to out-of-range magnitudes (Mw < 4.5 or Mw > 7.5).

**Keywords:** Artificial accelerograms; Variational autoencoder; Generative adversarial networks; Acceleration response spectra

---

## Features

- Conditional variational autoencoder–GAN (CVAE-GAN) framework for artificial accelerograms.
- Conditioning on **acceleration response spectra** for spectrum-compatible ground motions.
- Training data from **K-NET** and **KiK-net** strong-motion networks in Japan.
- P-wave onset detection and windowing based on **PhaseNet**.
- Evaluation in terms of response-spectrum similarity (R², RMSE) and time–frequency characteristics.
- Demonstrated generalization to magnitudes outside the training range.

---

## Data

- The model is trained on K-NET and KiK-net records.  
- Due to data usage policies, **raw waveforms are not redistributed in this repository**.  
- Users should download the strong-motion data directly from the official K-NET / KiK-net websites and follow their terms of use.
- Preprocessing (PhaseNet picking, baseline correction, filtering, response spectrum computation, etc.) follows the procedure described in the paper.

---

## Environment and Dependencies

This project is implemented in **Python** with deep learning libraries (e.g., TensorFlow/Keras or similar frameworks).

Typical requirements include:

- Python ≥ 3.8  
- NumPy, SciPy, Pandas  
- Matplotlib / Seaborn (for plotting)  
- TensorFlow / Keras (or the deep learning framework used in the scripts)  
- scikit-learn  
- h5py  
- PhaseNet or precomputed PhaseNet picks (if you want to reproduce the full preprocessing workflow)

Please install the required packages with `pip` or `conda` according to the versions used in your environment.

---

## Basic Usage

> The exact script names and arguments may differ; please adapt the examples below to your own file names.

1. **Training**

   ```bash
   python train_cvae_gan.py \
       --config configs/train_config.yaml
