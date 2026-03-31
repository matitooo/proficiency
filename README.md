# 🧠 English Proficiency Prediction with SenseFlow & Graph Neural Networks

**Course:** Mining Opinions and Arguments
**Academic Year:** 2025–2026
**Institution:** Universität Potsdam

**Authors:**

* Mattia D’Agostini
* Maria Manina
* Kwun Pok Wong

---

## Overview

This project focuses on **predicting English proficiency levels** using advanced machine learning techniques, combining:

* **SenseFlow representations**
* **Graph Neural Networks (GNNs)**

The goal is to model linguistic and semantic structures to improve classification performance on proficiency prediction tasks.

---

## Dataset

The data used in this project comes from the **Celva.Sp dataset**.

🔗 [https://hal.science/hal-04968220v1](https://hal.science/hal-04968220v1)

---

## Full Report

A detailed explanation of the methodology, experiments, and results is available here:

🔗 [https://github.com/matitooo/proficiency/tree/main/data/report.pdf](https://github.com/matitooo/proficiency/tree/main/data/report.pdf)

---

## Installation

The project was tested using **Python 3.11**.

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Usage

The project provides two main modes of operation:

### Training Mode

Train models using the best hyperparameters found during tuning:

```bash
python main.py --train --model <model_name>
```

### Hyperparameter Sweep Mode

Perform a **Bayesian hyperparameter optimization**:

```bash
python main.py --sweep --model <model_class>
```

---

## Model Classes

You can select a model class using the `--model` flag.

### 1. Linear Models

* MLP
* Decision Tree Classifier
* Random Forest Classifier
* Logistic Regression

---

### 2. Sequential Models

* BiLSTM
* Multi-Head Attention

---

### 3. Graph Models

* GCN (Graph Convolutional Network)
* GAT (Graph Attention Network)

---

### 4. Mixed Models

* BiLSTM combined with GAT (adapted for mixed inputs)

---

## Usage Example

Train a linear model:

```bash
python main.py --train --model linear
```

---



## 📬 Contact

For questions or collaboration, feel free to reach out to the authors.

---

