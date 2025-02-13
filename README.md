# Gravitational Wave Anomaly Detection using Normalised Autoencoders

This repository implements anomaly detection in gravitational waves using **normalised autoencoders**, inspired by the [Manifold Projection Diffusion Recovery (MPDR)](https://github.com/swyoon/manifold-projection-diffusion-recovery-pytorch) method. It provides scripts to train both standard and normalised autoencoders for detecting anomalies in gravitational wave signals.

---

## Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Directory Structure

```plaintext
mpdr_physics/
├── dataset/                # Contains dataset-related files
├── models/                 # Autoencoder and MPDR model definitions
├── train/                  # Training scripts and configurations
├── utils/                  # Helper functions for preprocessing and evaluation
├── train_ae.sh             # Train a standard autoencoder
├── train_mpdr-r.sh         # Train normalised autoencoder (MPDR-r version)
├── train_mpdr-r_best.sh    # Optimized MPDR-r training script
├── train_mpdr-r_optuna.sh  # MPDR-r with hyperparameter tuning (Optuna)
├── train_mpdr-s.sh         # Train MPDR-s version
├── train_mpdr-s_best.sh    # Optimized MPDR-s training script
├── train_mpdr-s_optuna.sh  # MPDR-s with Optuna tuning
├── train_netx.sh           # Train a energy autoencoder
├── requirements.txt        # Required dependencies
└── LICENSE                 # License file




