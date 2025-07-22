# Out-of-Context Multi-Modal CheapFakes Detection

A multimodal system for detecting **CheapFakes**—image-caption mismatches where the image remains untouched but the caption presents misleading context. This project combines **scene graph reasoning**, **deep visual-textual encoding**, and **natural language inference (NLI)** to determine whether one of two captions for the same image is contextually incorrect.

📌 **Thesis Project** – VNUHCM - University of Science, B.Sc. in Data Science (2020–2024)

---

## 🧠 Problem Overview

Out-of-context (OOC) misinformation is subtle but powerful: real images are paired with captions that distort their meaning. Unlike traditional deepfake detection, this task requires **semantic understanding between images and text**.

The system identifies inconsistencies between an image and two candidate captions using multimodal inputs, aiming to support fact-checking workflows.

---

## 🔍 Research Objectives

- Detect whether one of two captions describing the same image is contextually misleading.
- Design a model architecture that fuses visual and textual semantic graphs.
- Explore both **supervised learning** and **hybrid inference approaches**.
- Evaluate effectiveness on synthetic datasets built to simulate real-world OOC cases.

---

## 🧩 System Architecture

### 🖼 Visual Scene Graphs
- Extracted via **Neural Motifs** and **PE-Net**.
- Retain top **36 objects** and **25 relationships** per image.
- Scene graph represented as a tuple **G = (O, B, R)**:
  - `O`: object labels
  - `B`: bounding boxes
  - `R`: pairwise object relationships

### 📝 Textual Scene Graphs
- Generated using **SPICE**; fall back to **FACTUAL-T5** when parsing fails.
- Captions parsed into semantic graphs reflecting entity and relation structures.

### 🧠 Encoders
- **Visual Encoder**: EfficientNet-B5 + GCN with Swish activation
- **Textual Encoder**: LSTM over structured scene graphs
- Multimodal feature fusion supports joint reasoning.

### ⚖️ Natural Language Inference (NLI)
- Applied **DeBERTa-v3** and **RoBERTa** to caption pairs.
- Labels: `Entailment`, `Contradiction`, `Neutral`
- Used to strengthen contextual mismatch prediction.

---

## 📊 Dataset Construction

Four synthetic datasets (D1–D4) were created to simulate OOC scenarios:

| Dataset | Samples | Augmentation Techniques | OOC Types |
|---------|---------|-------------------------|-----------|
| D1      | 12,000  | Caption replacement     | OOC1      |
| D2      | 16,000  | Named entity replacement| OOC2      |
| D3      | 80,000  | Mix of above            | OOC1+OOC2 |
| D4      | 160,000 | + Claim injection       | OOC1+OOC2+OOC3 |

Augmentations include:
- Using semantically distant captions (SimCLR + cosine similarity)
- Replacing named entities with others via SpaCy NER
- Injecting fake-claim phrases ("This is fake news", etc.)

---

## 📈 Experimental Results

| Visual Graph | NLI Model    | D1     | D2     | D3     | D4     |
|--------------|--------------|--------|--------|--------|--------|
| PE-Net       | RoBERTa      | 0.751  | 0.777  | 0.793  | 0.819  |
| PE-Net       | DeBERTa-v3   | **0.796** | **0.804** | **0.823** | **0.834** |
| Neural Motifs| RoBERTa      | 0.761  | 0.787  | 0.791  | 0.802  |
| Neural Motifs| DeBERTa-v3   | 0.770  | 0.792  | 0.810  | 0.822  |

📌 **Best performance**: PE-Net + DeBERTa-v3 with **83.4% accuracy** on D4 dataset.

---

## 🎯 Key Takeaways

- Multimodal feature fusion with semantic graphs effectively detects OOC cases.
- NLI improves reasoning over caption contradictions.
- Augmented data enables scalable and realistic OOC training scenarios.

---

## 🛠 Tech Stack

- **Vision**: EfficientNet-B5, PE-Net, Neural Motifs
- **Text**: LSTM, SPICE, FACTUAL-T5
- **Graphs**: Scene Graphs, GCN
- **NLI**: DeBERTa-v3, RoBERTa
- **Frameworks**: PyTorch, SpaCy, Transformers

---


## Acknowledgements

- Based on methods from:
  - Neural Motifs (Zellers et al.)
  - PE-Net
  - EfficientNet (Tan & Le)
  - DeBERTa-v3 (Microsoft)
  - SPICE / FACTUAL-T5 for semantic parsing

## License


