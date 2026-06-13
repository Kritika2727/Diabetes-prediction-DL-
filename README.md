# Diabetes-prediction-DL
# 🧠 Diabetes Prediction — Neural Network (Deep Learning)

A deep learning approach to diabetes prediction using a multi-layer Neural Network built with Keras and TensorFlow. This is the **Part 2** companion to the [Classical ML project](https://github.com/Kritika2727/Diabetes-prediction-ML-) — same dataset, deeper approach.

---

## 📌 Why Neural Network After Classical ML?

Classical ML models (Logistic Regression, Random Forest) gave ~77% accuracy. This project asks: *Can a deep neural network do better — and more importantly, can it handle class imbalance more effectively?*

The answer isn't just accuracy. It's in the ROC-AUC and Precision-Recall curves.

---

## 📊 Dataset

- **Source:** Pima Indians Diabetes Dataset
- **Features:** Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
- **Target:** Outcome (0 = Non-Diabetic, 1 = Diabetic)
- **Size:** 768 records
- **Challenge:** Imbalanced classes — more non-diabetic than diabetic cases

---

## 🏗️ Neural Network Architecture

```
Input Layer  →  8 features
Dense(256)   →  ReLU activation
Dense(128)   →  ReLU activation
Dense(64)    →  ReLU activation
Dense(32)    →  ReLU activation
Dense(1)     →  Sigmoid activation (binary output)

Optimizer: Adam
Loss: Binary Crossentropy
Callback: EarlyStopping (patience=10, monitors val_accuracy)
```

---

## 🔧 What This Project Covers

### 1. Data Cleaning & Preprocessing
- Zero value imputation using skewness-based mean/median replacement
- Feature scaling with StandardScaler
- Train-test split (90/10)

### 2. Exploratory Data Analysis
- Correlation heatmap
- Feature distribution histograms (all 8 features)
- Box plots by diabetes outcome

### 3. Class Imbalance Handling with SMOTE
- Applied SMOTE only on training data (correct approach — no data leakage)
- Balanced class distribution before training

### 4. Two Models Trained & Compared
| Model | Accuracy | F1 (Diabetic) |
|---|---|---|
| Neural Network (Without SMOTE) | 66% | 0.57 |
| Neural Network (With SMOTE) | 71% | 0.58 |

### 5. Advanced Evaluation
- **Precision-Recall Curve** — best metric for imbalanced medical data
- **ROC Curve with AUC Score** — overall discriminatory power
- Side-by-side comparison: SMOTE vs Non-SMOTE on both curves

---

## 📈 Key Insight

> ROC-AUC tells you how well a model separates classes overall. Precision-Recall tells you how well it handles the minority class specifically. For diabetes screening — where missing a diabetic case is more dangerous than a false alarm — **Precision-Recall matters more than ROC-AUC.**

SMOTE improved overall accuracy from 66% → 71% and macro-average F1 score — but came with a slight recall trade-off for diabetic cases. This trade-off is visible in the Precision-Recall curve and is a critical finding for medical ML applications.

---

## 🛠️ Tools & Libraries

- Python
- TensorFlow & Keras
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- imbalanced-learn (SMOTE)
- Google Colab

---

## 📁 Project Structure

```
Diabetes-Prediction-Neural-Network/
│
├── diabetesNN.ipynb        # Main notebook
├── README.md               # Project documentation
```

---

## 🔗 Related Project

**Part 1 — Classical ML Approach:**
[Diabetes Prediction using Logistic Regression, Decision Tree & Random Forest](https://github.com/Kritika2727/Diabetes-prediction-ML-)

Same dataset. Different approach.

---
