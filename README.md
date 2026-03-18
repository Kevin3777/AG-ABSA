# Beyond Cosine Similarity: Zero-Initialized Residual Complex Projection for Aspect-Based Sentiment Analysis

[![arXiv](https://img.shields.io/badge/arXiv-2303.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2303.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-blue)](https://huggingface.co/Kevin3777/zero-init-residual-complex-absa)

Official PyTorch implementation of the paper:  
**"BEYOND COSINE SIMILARITY: ZERO-INITIALIZED RESIDUAL COMPLEX PROJECTION FOR ASPECT-BASED SENTIMENT ANALYSIS"**  
*March 17, 2026*

This repository contains the code, data, and pre-trained models for our novel framework that disentangles aspect semantics and sentiment polarities using complex-valued representations. Our method achieves state-of-the-art performance on fine-grained ABSA tasks by overcoming false-negative collisions and leveraging phase-driven angle optimization.

---

## 📖 Table of Contents

- [Overview](#overview)
- [Key Contributions](#key-contributions)
- [Methodology](#methodology)
  - [Zero-Initialized Residual Complex Projection (ZRCP)](#zero-initialized-residual-complex-projection-zrcp)
  - [Anti-collision Masked Angle Loss](#anti-collision-masked-angle-loss)
  - [Joint Optimization Objective](#joint-optimization-objective)
- [Experiments](#experiments)
  - [Dataset](#dataset)
  - [Baselines](#baselines)
  - [Main Results](#main-results)
  - [Ablation Study](#ablation-study)
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
- Introduces an **Anti-collision Masked Angle Loss** that prevents false-negative repulsion and expands inter-polarity margins by over 50%.

Our method achieves a **Macro-F1 of 0.8851** on the ASAP dataset, significantly outperforming strong baselines.

The pre-trained model is available on Hugging Face: [Kevin3777/zero-init-residual-complex-absa](https://huggingface.co/Kevin3777/zero-init-residual-complex-absa)

---

## 🏆 Key Contributions

1. **ZRCP Module**: A residual complex projection that smoothly maps pre-trained real-valued features into a complex space, preserving semantic continuity while decoupling aspect and sentiment.
2. **Anti-collision Masked Angle Loss**: A novel loss that dynamically masks same-polarity samples to avoid false negatives and maximizes angular divergence between opposing sentiments.
3. **Geometric Analysis**: We demonstrate that explicitly penalizing amplitude leads to over-regularization, proving that amplitude naturally captures subjective intensity and must remain unconstrained.
4. **State-of-the-Art Results**: Extensive experiments on 18 fine-grained aspects show consistent improvements over existing methods.

---

## ⚙️ Methodology

### Zero-Initialized Residual Complex Projection (ZRCP)

Instead of hard chunking the hidden state into real/imaginary parts (which destroys semantic integrity), we use two zero-initialized linear projections with residual connections:

$$
h_{\text{re}} = \mathrm{chunk}(h)_{\text{re}} + W_{\text{re}}(\mathrm{chunk}(h)_{\text{re}} + b_{\text{re}})
$$

$$
h_{\text{im}} = \mathrm{chunk}(h)_{\text{im}} + W_{\text{im}}(\mathrm{chunk}(h)_{\text{im}} + b_{\text{im}})
$$

At initialization, ZRCP acts as an identity mapping; during training, it learns to disentangle aspect (real part) and polarity (imaginary part) without spatial collapse.

### Anti-collision Masked Angle Loss

Standard contrastive loss erroneously repels same-polarity samples. We introduce a mask matrix \(M_{ij}\) that zeros out the penalty for pairs sharing the same aspect and polarity:

$$
\mathcal{L}_{\text{ibn}} = -\sum_{i=1}^{N} \log \left( \frac{e^{\mathrm{sim}(h_i,h_i^+)\tau_{\text{ibn}}}}{\sum_{j=1}^{N} e^{\mathrm{sim}(h_i,h_j)\tau_{\text{ibn}}} \cdot (1 - M_{ij})} \right)
$$

### Angle-Optimized Objective

We explicitly maximize the angular divergence between opposing sentiments using complex division and amplitude normalization:

$$
\mathcal{L}_{\text{angle}} = \log\left(1 + \sum_{n\in \mathcal{N}} \exp\left(\Delta \theta_{zn}\tau_{\text{angle}} - \Delta \theta_{zw}\tau_{\text{angle}}\right)\right)
$$

### Joint Loss

$$
\mathcal{L}_{\text{total}} = w_{\text{ibn}}\mathcal{L}_{\text{ibn}} + w_{\text{angle}}\mathcal{L}_{\text{angle}} + w_{\text{cos}}\mathcal{L}_{\text{cos}}
$$

In our optimal configuration: $w_{\text{ibn}}=1.0$, $w_{\text{angle}}=1.0$, $w_{\text{cos}}=0.1$, $\tau_{\text{ibn}}=\tau_{\text{angle}}=20$.

---


## 📦 Dataset

### ASAP-Triplet Dataset

We release the pre-processed triplet dataset used in our experiments to facilitate future research.

**Dataset Description**:
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

### Dataset

We evaluate on the **ASAP** (Aspect-based Sentiment Analysis of Restaurant Reviews) dataset, which contains Chinese restaurant reviews annotated with 18 fine-grained aspects (e.g., Location, Service, Price, Taste) and sentiment polarities (Positive, Negative, Neutral). After context-aware extraction, we obtain purified aspect-polarity triplets.

### Baselines

- **RoBERTa (Zero-shot)**: Pre-trained Chinese RoBERTa without fine-tuning.
- **CoSENT**: Cosine-based sentence embedding method.
- **SimCSE**: Contrastive learning with dropout augmentation.
- **AnglE (Format C)**: Original angle-optimized embeddings with hard chunking and standard triplets.

### Main Results

| Aspect Category   | RoBERTa (Zero-shot) | CoSENT       | SimCSE       | AnglE (Format C) | Ours (ZRCP+Mask) |
|-------------------|---------------------|--------------|--------------|------------------|------------------|
|                   | F1     | Acc     | F1     | Acc     | F1     | Acc     | F1     | Acc     | F1     | Acc     |
| Transportation    | 0.6291 | 0.8400  | 0.7972 | 0.9356  | 0.8517 | 0.9549  | 0.8409 | 0.9549  | **0.8790** | **0.9635** |
| Downtown          | 0.5715 | 0.9257  | 0.6890 | 0.9575  | 0.7663 | 0.9729  | **0.8171** | 0.9800  | 0.7372 | 0.9717 |
| Easy to find      | 0.6787 | 0.7362  | 0.8525 | 0.8906  | 0.9189 | 0.9435  | 0.9100 | 0.9366  | **0.9382** | **0.9574** |
| Queue             | 0.7332 | 0.7340  | 0.7523 | 0.7532  | 0.8206 | 0.8213  | 0.8118 | 0.8128  | **0.8413** | **0.8426** |
| Hospitality       | 0.8607 | 0.8984  | 0.8807 | 0.9134  | 0.9393 | 0.9583  | 0.9405 | 0.9592  | **0.9429** | **0.9610** |
| Parking           | 0.6156 | 0.6693  | 0.7704 | 0.8132  | 0.8798 | 0.9026  | 0.8674 | 0.8912  | **0.8908** | **0.9106** |
| Timely            | 0.7413 | 0.7508  | 0.8678 | 0.8723  | 0.9172 | 0.9210  | **0.9366** | **0.9392** | 0.9330 | 0.9362 |
| Price Level       | 0.7427 | 0.7433  | 0.8049 | 0.8056  | 0.9193 | 0.9199  | 0.9194 | 0.9199  | **0.9223** | **0.9228** |
| Cost-effective    | 0.7439 | 0.8461  | 0.8117 | 0.8981  | 0.9431 | 0.9734  | 0.9223 | 0.9628  | **0.9459** | **0.9745** |
| Discount          | 0.7012 | 0.8252  | 0.6993 | 0.8365  | 0.7251 | 0.8692  | 0.7744 | 0.9006  | **0.7893** | **0.9094** |
| Decoration        | 0.6815 | 0.8727  | 0.7004 | 0.8975  | 0.8181 | 0.9487  | 0.8281 | 0.9539  | **0.8494** | **0.9608** |
| Noise             | 0.7272 | 0.8385  | 0.7990 | 0.8884  | 0.9084 | 0.9550  | **0.9306** | **0.9667** | 0.9294 | 0.9650 |
| Space             | 0.7431 | 0.8003  | 0.8104 | 0.8638  | 0.8563 | 0.9001  | 0.8774 | 0.9160  | **0.8941** | **0.9266** |
| Sanitary          | 0.7705 | 0.8535  | 0.8094 | 0.8803  | 0.8694 | 0.9254  | 0.8744 | 0.9296  | **0.8857** | **0.9359** |
| Portion           | 0.7220 | 0.7617  | 0.8170 | 0.8487  | 0.8719 | 0.8968  | 0.8776 | 0.9020  | **0.8990** | **0.9194** |
| Taste             | 0.8486 | 0.9504  | 0.8651 | 0.9566  | 0.8966 | 0.9682  | 0.8973 | 0.9685  | **0.9139** | **0.9747** |
| Appearance        | 0.7234 | 0.8373  | 0.7426 | 0.8532  | 0.8171 | 0.9069  | 0.8366 | 0.9182  | **0.8327** | **0.9144** |
| Recommend         | 0.6695 | 0.8023  | 0.7814 | 0.8941  | 0.8882 | 0.9515  | 0.8825 | 0.9490  | **0.9079** | **0.9605** |
| **Macro-Average** | 0.7169 | 0.8437  | 0.7917 | 0.8755  | 0.8671 | 0.9272  | 0.8756 | 0.9315  | **0.8851** | **0.9393** |

**Table 1:** Main results on the ASAP dataset. Best scores are in bold.

### Ablation Study

| Model Variations          | Macro-F1 | Acc    |
|---------------------------|----------|--------|
| Ours (Full Model)         | **0.8851** | **0.9393** |
| w/o ZRCP (Hard Chunking)  | 0.8739   | 0.9319 |
| w/o Anti-collision Mask   | 0.8749   | 0.9308 |
| w/o Angle Loss (\(w_{\text{angle}}=0\)) | 0.8843   | 0.9389 |

**Table 2:** Ablation study on core components.

### Amplitude Penalty Analysis

We investigate the effect of adding an explicit amplitude penalty (MSE loss) to force identical magnitudes for same-aspect samples.

| Aspect Category   | w/o Amploss (Ours) | with Amploss | Δ        |
|-------------------|--------------------|--------------|----------|
| **Objective / Fact-based**   |                    |              |          |
| Discount          | 0.7893             | 0.7983       | +0.90%   |
| Price Level       | 0.9223             | 0.9261       | +0.38%   |
| Timely            | 0.9330             | 0.9346       | +0.16%   |
| **Subjective / Descriptive** |                    |              |          |
| Decoration        | 0.8494             | 0.8403       | -0.91%   |
| Taste             | 0.9139             | 0.9095       | -0.44%   |
| Appearance        | 0.8327             | 0.8286       | -0.41%   |
| **Macro-Average** | 0.8851             | 0.8829       | -0.22%   |

**Table 3:** Impact of amplitude penalty. Amplitude regularization helps factual aspects but harms subjective ones, confirming that amplitude encodes sentiment intensity.

---

## 🚀 Getting Started

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/ZRCP-ABSA.git
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
model = AutoModel.from_pretrained("yourusername/zrcp-absa")
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
    "input_jsonl_file": "D:/WorkSpace/AnglE_yj/data_preparation/Aspect-Polarity_Pair/output/v2/asap_angle_contextual_ap_data_hybrid.jsonl",
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
@article{yourname2026beyond,
  title={Beyond Cosine Similarity: Zero-Initialized Residual Complex Projection for Aspect-Based Sentiment Analysis},
  author={Your Name and Collaborators},
  journal={arXiv preprint arXiv:2303.XXXXX},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

We thank the authors of the ASAP dataset and the open-source community for their valuable contributions.

---

**Note:** The code and data will be released upon paper acceptance. For questions, please open an issue or contact [your.email@example.com](mailto:your.email@example.com).
