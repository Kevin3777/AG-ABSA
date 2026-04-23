# Beyond Cosine Similarity: Zero-Initialized Residual Complex Projection for Aspect-Based Sentiment Analysis

[![arXiv](https://img.shields.io/badge/arXiv-2603.28205-b31b1b.svg)](https://arxiv.org/abs/2603.28205)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-blue)](https://huggingface.co/Kevin3777/zero-init-residual-amp)

Official PyTorch implementation of the paper:  
**"BEYOND COSINE SIMILARITY: ZERO-INITIALIZED RESIDUAL COMPLEX PROJECTION FOR ASPECT-BASED SENTIMENT ANALYSIS"**  
*March 17, 2026*

This repository contains the code, data, and pre-trained models for our novel framework that disentangles aspect semantics and sentiment polarities using complex-valued representations. Our method achieves state-of-the-art performance on fine-grained ABSA tasks by overcoming false-negative collisions and leveraging phase-driven angle optimization, further consolidated by an **amplitude penalty** that filters subjective intensity noise.

---

## 📖 Table of Contents

- [Overview](#overview)
- [Key Contributions](#key-contributions)
- [Methodology](#methodology)
  - [Zero-Initialized Residual Complex Projection (ZRCP)](#zero-initialized-residual-complex-projection-zrcp)
  - [Anti-collision Masked Angle Loss](#anti-collision-masked-angle-loss)
  - [Amplitude Penalty for Structural Consolidation](#amplitude-penalty-for-structural-consolidation)
  - [Joint Optimization Objective](#joint-optimization-objective)
- [Experiments](#experiments)
  - [Datasets](#datasets)
  - [Baselines](#baselines)
  - [Main Results](#main-results)
  - [Ablation Study](#ablation-study)
  - [English Benchmarks](#english-benchmarks)
  - [Amplitude Penalty Analysis](#amplitude-penalty-analysis)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Data Preparation](#data-preparation)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Citation](#citation)
- [License](#license)

---

## 🔍 Overview

Aspect-Based Sentiment Analysis (ABSA) suffers from **representation entanglement** – aspect semantics and sentiment polarities are often conflated in real-valued embedding spaces. Moreover, standard contrastive learning introduces **false-negative collisions**, pushing apart sentences that share the same aspect and polarity, which destroys intra-aspect cohesion.

We propose a novel framework that:
- Projects textual features into a **complex semantic space** using a **Zero-Initialized Residual Complex Projection (ZRCP)** module.
- Uses the **phase** (angle) to represent sentiment polarity and the **amplitude** (magnitude) to encode semantic intensity and lexical richness.
- Introduces an **Anti-collision Masked Angle Loss** that prevents false-negative repulsion and expands inter-polarity margins.
- Further consolidates representations with an **Amplitude Penalty** that constrains same-aspect vectors onto a unified hypersphere, effectively filtering out intensity-related noise.

Our method achieves a **Macro-F1 of 0.8923** and **Accuracy of 0.9418** on the ASAP dataset, significantly outperforming strong baselines. It also achieves state-of-the-art results on English benchmarks SemEval-2016 (Macro-F1 0.9044) and MAMS-ACSA (Macro-F1 0.6671).

The pre-trained model is available on Hugging Face: [Kevin3777/zero-init-residual-complex-absa](https://huggingface.co/Kevin3777/zero-init-residual-amp)

---

## 🏆 Key Contributions

1. **ZRCP Module**: A zero-initialized residual complex projection that smoothly maps pre-trained real-valued features into a complex space, preserving semantic continuity while decoupling aspect and sentiment.
2. **Anti-collision Masked Angle Loss**: A novel loss that dynamically masks same-polarity samples to avoid false negatives and maximizes angular divergence between opposing sentiments.
3. **Amplitude Penalty (L_amp)**: A structural regularizer that forces same-aspect representations to reside on a consistent hypersphere, filtering out subjective intensity noise and significantly boosting performance on long-tail aspects (e.g., +3.69% F1 on Downtown).
4. **State-of-the-Art Results**: Extensive experiments on Chinese ASAP (18 fine-grained aspects) and two English benchmarks show consistent and substantial improvements over existing methods.

---

## ⚙️ Methodology

### Zero-Initialized Residual Complex Projection (ZRCP)

Instead of hard chunking the hidden state into real/imaginary parts (which destroys semantic integrity), we use two zero-initialized linear projections with residual connections:

$$
h_{\text{re}} = h^{(1)} + (W_{\text{re}} h^{(1)} + b_{\text{re}})
$$

$$
h_{\text{im}} = h^{(2)} + (W_{\text{im}} h^{(2)} + b_{\text{im}})
$$

where $h^{(1)}, h^{(2)}$ are equal splits of the original PLM output. All weights and biases are initialized to zero, so at initialization ZRCP acts as an identity mapping, preserving pre-trained knowledge while learning smooth phase shifts for polarity disentanglement.

### Anti-collision Masked Angle Loss

Standard contrastive loss erroneously repels same-polarity samples. We introduce a mask matrix $M_{ij}$ that zeros out the penalty for pairs sharing the same aspect and polarity:

$$
\mathcal{L}_{\text{ibn}} = -\sum_{i=1}^{N} \log \left( \frac{e^{\text{sim}(h_i,h_i^+)\tau_{\text{ibn}}}}{\sum_{j=1}^{N} e^{\text{sim}(h_i,h_j)\tau_{\text{ibn}}} \cdot (1 - M_{ij})} \right)
$$

Additionally, we directly optimize the complex angular difference to overcome gradient saturation of cosine similarity:

$$
\mathcal{L}_{\text{angle}} = \log\left(1 + \sum_{n\in \mathcal{N}} \exp\left(\Delta \theta_{zn}\tau_{\text{angle}} - \Delta \theta_{zw}\tau_{\text{angle}}\right)\right)
$$

### Amplitude Penalty for Structural Consolidation

To enforce structural consistency within the same aspect category, we add an amplitude penalty that aligns the complex magnitudes of query and target (regardless of polarity):

$$
\mathcal{L}_{\text{amp}} = \text{MSE}(|z|, |w|)
$$

This regularization filters out intensity noise (e.g., lexical richness) and forces the model to focus on phase-driven polarity separation, particularly benefiting class-imbalanced aspects.

### Joint Optimization Objective

$$
\mathcal{L}_{\text{total}} = w_{\text{ibn}}\mathcal{L}_{\text{ibn}} + w_{\text{angle}}\mathcal{L}_{\text{angle}} + w_{\text{cos}}\mathcal{L}_{\text{cos}} + w_{\text{amp}}\mathcal{L}_{\text{amp}}
$$

In our optimal configuration: $w_{\text{ibn}}=w_{\text{angle}}=w_{\text{cos}}=w_{\text{amp}}=1.0$, $\tau_{\text{ibn}}=\tau_{\text{angle}}=20$.

---

## 📦 Datasets

### ASAP-Triplet Dataset (Chinese)

We release the pre-processed triplet dataset used in our experiments.

- **Base dataset**: ASAP ([Aspect-based Sentiment Analysis of Restaurant Reviews](https://github.com/Meituan-Dianping/asap.git))
- **Language**: Chinese
- **Aspects**: 18 fine-grained categories (Location, Service, Price, Taste, etc.)
- **Polarities**: Positive, Negative, Neutral
- **Format**: JSON lines with triplet structure for contrastive learning

**Access**:
```bash
# Download from Hugging Face
git clone https://huggingface.co/datasets/Kevin3777/ASAP-Triplet

# Or load directly with datasets library
from datasets import load_dataset
dataset = load_dataset("Kevin3777/ASAP-Triplet")
```

## 📊 Experiments

### Baselines

- **RoBERTa (Zero-shot)**: Pre-trained Chinese RoBERTa without fine-tuning.
- **RoBERTa-pair**: Linear probe with aspect-context paired prompts.
- **LCFS-RoBERTa**: Local context-focused model.
- **DualGCN-RoBERTa**: Syntax-aware dual graph convolutional network.
- **SimCSE**: Contrastive learning with dropout augmentation.
- **AnglE (Format C/A)**: Original angle-optimized embeddings with hard chunking.

### Main Results (ASAP Dataset)

| Aspect Category | RoBERTa (Zero-shot) | RoBERTa-pair | LCFS | DualGCN | SimCSE | AnglE (Form C) | **Ours (ZRCP+Full)** |
|----------------|---------------------|--------------|------|---------|--------|----------------|----------------------|
|                | F1 / Acc            | F1 / Acc     | F1 / Acc | F1 / Acc | F1 / Acc | F1 / Acc | F1 / Acc |
| Transportation | 0.5359 / 0.6950 | 0.6559 / 0.8571 | 0.6834 / 0.8861 | 0.7950 / 0.9323 | 0.9039 / 0.9710 | 0.8622 / 0.9592 | **0.9022 / 0.9721** |
| Downtown       | 0.4706 / 0.7217 | 0.5700 / 0.9245 | 0.6084 / 0.9363 | 0.7454 / 0.9658 | 0.7961 / 0.9682 | 0.8434 / 0.9776 | **0.7858 / 0.9717** |
| Easy to find   | 0.5998 / 0.6601 | 0.7150 / 0.7696 | 0.7534 / 0.8076 | 0.9004 / 0.9297 | 0.9191 / 0.9424 | 0.9149 / 0.9401 | **0.9265 / 0.9482** |
| Queue          | 0.6830 / 0.6851 | 0.7502 / 0.7511 | 0.7649 / 0.7660 | 0.7802 / 0.7809 | 0.8401 / 0.8404 | 0.8573 / 0.8574 | **0.8743 / 0.8745** |
| Hospitality    | 0.7932 / 0.8526 | 0.8678 / 0.9039 | 0.8793 / 0.9138 | 0.9281 / 0.9497 | 0.9505 / 0.9660 | 0.9479 / 0.9642 | **0.9582 / 0.9714** |
| Parking        | 0.6495 / 0.6809 | 0.6570 / 0.7121 | 0.6720 / 0.7354 | 0.8319 / 0.8599 | 0.9072 / 0.9261 | 0.8965 / 0.9183 | **0.9169 / 0.9339** |
| Timely         | 0.6580 / 0.6733 | 0.8146 / 0.8222 | 0.8210 / 0.8283 | 0.8816 / 0.8860 | 0.9161 / 0.9195 | 0.9414 / 0.9438 | **0.9329 / 0.9362** |
| Price Level    | 0.6645 / 0.6662 | 0.7628 / 0.7634 | 0.7824 / 0.7826 | 0.9015 / 0.9021 | 0.9312 / 0.9318 | 0.9335 / 0.9340 | **0.9342 / 0.9347** |
| Cost-effective | 0.6600 / 0.7696 | 0.7522 / 0.8514 | 0.7689 / 0.8684 | 0.8732 / 0.9352 | 0.9035 / 0.9512 | 0.9223 / 0.9628 | **0.9212 / 0.9607** |
| Discount       | 0.5751 / 0.6994 | 0.6975 / 0.8264 | 0.7248 / 0.8604 | 0.7146 / 0.8453 | 0.7347 / 0.8654 | 0.7744 / 0.9006 | **0.7835 / 0.8956** |
| Decoration     | 0.5841 / 0.7759 | 0.6891 / 0.8808 | 0.7334 / 0.9113 | 0.7814 / 0.9343 | 0.8057 / 0.9412 | 0.8281 / 0.9539 | **0.8627 / 0.9637** |
| Noise          | 0.6127 / 0.7136 | 0.7539 / 0.8593 | 0.7714 / 0.8751 | 0.8840 / 0.9417 | 0.9079 / 0.9550 | 0.9306 / 0.9667 | **0.9272 / 0.9642** |
| Space          | 0.6557 / 0.7050 | 0.7695 / 0.8245 | 0.7710 / 0.8306 | 0.8805 / 0.9160 | 0.8809 / 0.9168 | 0.8774 / 0.9160 | **0.9108 / 0.9380** |
| Sanitary       | 0.7133 / 0.8106 | 0.7862 / 0.8641 | 0.8072 / 0.8838 | 0.8776 / 0.9289 | 0.8654 / 0.9197 | 0.8744 / 0.9296 | **0.9002 / 0.9423** |
| Portion        | 0.6490 / 0.6916 | 0.7347 / 0.7710 | 0.7690 / 0.8052 | 0.8531 / 0.8788 | 0.8864 / 0.9072 | 0.8776 / 0.9020 | **0.9013 / 0.9200** |
| Taste          | 0.7500 / 0.9001 | 0.8538 / 0.9522 | 0.8896 / 0.9667 | 0.8809 / 0.9624 | 0.9048 / 0.9710 | 0.8973 / 0.9685 | **0.9247 / 0.9776** |
| Appearance     | 0.6823 / 0.7996 | 0.7224 / 0.8363 | 0.7390 / 0.8561 | 0.7721 / 0.8692 | 0.8022 / 0.8899 | 0.8366 / 0.9182 | **0.8105 / 0.8975** |
| Recommend      | 0.6094 / 0.7194 | 0.6913 / 0.8189 | 0.7187 / 0.8444 | 0.7951 / 0.8941 | 0.8911 / 0.9515 | 0.8825 / 0.9490 | **0.8887 / 0.9503** |
| **Macro-Average** | 0.6414 / 0.7344 | 0.7358 / 0.8327 | 0.7588 / 0.8532 | 0.8376 / 0.9062 | 0.8748 / 0.9297 | 0.8823 / 0.9340 | **0.8923 / 0.9418** |

*Table: Main results on the ASAP dataset. Best scores are in bold.*

### Ablation Study

| Model Variations | Macro-F1 | Acc |
|------------------|----------|-----|
| Ours (Full Model) | **0.8923** | **0.9418** |
| w/o ZRCP (Hard Chunking) | 0.8739 | 0.9319 |
| w/o Anti-collision Mask | 0.8809 | 0.9336 |
| w/o Angle Loss (w_angle=0) | 0.8843 | 0.9389 |
| w/o Dynamic Window (Full Text) | 0.8679 | 0.9283 |
| w/o Hybrid Triplets (Standard Pairs) | 0.7735 | 0.8607 |

### English Benchmarks

| Model | SemEval-2016 (Macro-F1 / Acc) | MAMS-ACSA (Macro-F1 / Acc) |
|-------|-------------------------------|-----------------------------|
| RoBERTa-pair | 0.8180 / 0.8869 | 0.6383 / 0.7193 |
| DualGCN | 0.8289 / 0.9135 | 0.5651 / 0.6655 |
| SimCSE | 0.8457 / 0.8962 | 0.6038 / 0.7217 |
| AnglE (Format C) | 0.8632 / 0.9134 | — |
| AnglE (Format A) | 0.8784 / 0.9078 | 0.6532 / 0.7526 |
| **Ours (ZRCP+Full)** | **0.9044 / 0.9347** | **0.6671 / 0.7656** |

### Amplitude Penalty Analysis

Adding the amplitude penalty ($\mathcal{L}_{\text{amp}}$) improves Macro-F1 from 0.8901 to 0.8923, with especially large gains on long-tail aspects (e.g., Downtown +3.69%). It acts as a structural regularizer, filtering intensity noise without harming overall accuracy.

| Aspect Category | w/o AmpLoss | with AmpLoss | Δ |
|----------------|-------------|--------------|---|
| Transportation | 0.8916 | 0.9022 | +1.06% |
| Downtown       | 0.7489 | 0.7858 | +3.69% |
| Decoration     | 0.8556 | 0.8627 | +0.71% |
| Taste          | 0.9265 | 0.9247 | -0.18% |
| **Macro-F1**   | 0.8901 | **0.8923** | +0.22% |

---

## 🚀 Getting Started

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Kevin3777/AG-ABSA.git
cd AG-ABSA
pip install -r requirements.txt
```

### Data Preparation

We provide the pre-processed triplet dataset on Hugging Face Datasets, which only contains the training data for the triplet group:

```bash
# Option 1: Clone the dataset repository
git clone https://huggingface.co/datasets/Kevin3777/ASAP-Triplet
cp asap-triplet/*.json ./data/
```

You could have the original datasets including the train, dev and test data from https://github.com/Meituan-Dianping/asap.git

```bash
# Option 1: Clone the dataset repository
git clone https://huggingface.co/datasets/Kevin3777/ASAP-Triplet
cp asap-triplet/*.json ./data/
```

### Direct Usage

The pre-trained model is available on Hugging Face: [Kevin3777/zero-init-residual-complex-absa](https://huggingface.co/Kevin3777/zero-init-residual-complex-absa) To use the pre-trained model with codes:

```bash
from transformers import AutoModel
model = AutoModel.from_pretrained("Kevin3777/zero-init-residual-amp")
```

### Training

To train the model with default settings:

```bash
python train_learnable\v3_all1\train_encoder.py
```

The training config is shown as an example:

```bash
{
  "data": {
    "input_jsonl_file": "data_preparation/Aspect-Polarity_Pair/output/v2/asap_angle_contextual_ap_data_hybrid.jsonl",
    "output_dir": "checkpoints_learnable/v3_all1"
  },
  "model": {
    "name": "hfl/chinese-roberta-wwm-ext",
    "max_length": 192,
    "pooling_strategy": "cls"
  },
  "training": {
    "batch_size": 32,
    "gradient_accumulation_steps": 4,
    "num_epochs": 5,
    "learning_rate": 2e-5,
    "warmup_steps": 500,
    "save_steps": 1000,
    "logging_steps": 100,
    "fp16": true
  },
  "loss": {
    "cosine_w": 1.0,       
    "ibn_w": 1.0,
    "angle_w": 1.0,
    "cosine_tau": 20,
    "ibn_tau": 20,
    "angle_tau": 1          
  }
}

```

### Evaluation

Evaluate a trained model on the test set:

```bash
python eval_new\learnable\v3_all1\eval_learnable.py
```

---

## 📝 Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@misc{wang2026cosinesimilarityzeroinitializedresidual,
      title={Beyond Cosine Similarity: Zero-Initialized Residual Complex Projection for Aspect-Based Sentiment Analysis}, 
      author={Yijin Wang and Fandi Sun},
      year={2026},
      eprint={2603.28205},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.28205}, 
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

We thank the authors of the ASAP dataset and the open-source community for their valuable contributions.

---

**Note:** The code and data will be released upon paper acceptance. For questions, please open an issue or contact [wyj3777@outlook.com](mailto:wyj3777@outlook.com).
