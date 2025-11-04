# 💲 Fraud Detection Using Machine Learning

A practical machine learning project that detects fraudulent financial transactions in real-time.
This project demonstrates how advanced classification algorithms can significantly reduce financial fraud by identifying suspicious activities with high accuracy.

---

## 📌 Project Overview

Fraudulent activities in digital payments and mobile banking systems can lead to severe financial losses.
This project builds and evaluates multiple machine learning models to automatically flag potentially fraudulent transactions based on behavioral and financial patterns.

**Objectives:**

* Build a robust supervised ML model for fraud detection.
* Compare multiple algorithms for predictive performance.
* Assess the financial and operational impact of the model.

---

## 🧠 Machine Learning Workflow

### 1. Data Exploration (EDA)

* Analyzed **`Fraud_Analysis_Dataset.csv`** to understand data distribution and imbalance.
* Investigated transaction types (`CASH-IN`, `CASH-OUT`, `TRANSFER`, etc.).
* Visualized fraud proportions and correlations using Matplotlib and Seaborn.

### 2. Feature Engineering

* Encoded categorical variables (`type`).
* Scaled continuous features using `StandardScaler`.
* Derived additional features such as transaction deltas and ratios to improve fraud signal strength.

### 3. Model Development

Trained and compared the following models:

* **Logistic Regression**
* **Decision Tree Classifier**
* **Random Forest Classifier**
* **Gradient Boosting**
* **XGBoost**

### 4. Model Evaluation

Evaluated models using key classification metrics:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**
* **ROC-AUC Score**

Emphasis was placed on **precision** (to reduce false positives) and **recall** (to minimize undetected fraud).

---

## 📊 Model Performance Comparison

| Model               | Accuracy   | Precision  | Recall     | F1-Score   | ROC-AUC    |
| ------------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Logistic Regression | 0.9861     | 0.9950     | 0.8684     | 0.9274     | 0.9724     |
| Decision Tree       | 0.9960     | 0.9867     | 0.9737     | 0.9801     | 0.9861     |
| Random Forest       | 0.9960     | 1.0000     | 0.9605     | 0.9799     | 0.9999     |
| Gradient Boosting   | 0.9964     | 1.0000     | 0.9649     | 0.9821     | 0.9996     |
| **XGBoost**         | **0.9982** | **1.0000** | **0.9825** | **0.9912** | **1.0000** |

✅ **XGBoost** achieved the best overall performance with a near-perfect ROC-AUC of **1.0000**.

---

## ⚙️ Installation & Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/fraud-detection-ml.git
cd fraud-detection-ml
pip install -r requirements.txt
```

Or open directly in Jupyter Notebook:

```bash
jupyter notebook "Capstone Project - Fraud Detection.ipynb"
```

---

## 🚀 Usage

1. Place `Fraud_Analysis_Dataset.csv` in the project root.
2. Run the notebook to:

   * Explore and preprocess data.
   * Train models and evaluate results.
   * Visualize fraud distribution and model performance.

---

## 📂 Dataset Details

| Column           | Description                                                |
| ---------------- | ---------------------------------------------------------- |
| `step`           | Time unit (1 step = 1 hour)                                |
| `type`           | Transaction type (`CASH-IN`, `CASH-OUT`, `TRANSFER`, etc.) |
| `amount`         | Transaction amount                                         |
| `oldbalanceOrg`  | Sender balance before transaction                          |
| `newbalanceOrig` | Sender balance after transaction                           |
| `oldbalanceDest` | Receiver balance before transaction                        |
| `newbalanceDest` | Receiver balance after transaction                         |
| `isFraud`        | 1 = fraudulent, 0 = legitimate                             |

---

## 🔍 Key Insights

* **Fraud occurs primarily in ****`TRANSFER`**** and ****`CASH-OUT`**** transactions.**
* **Data imbalance handling** (fraud <1%) was crucial for achieving high recall.
* **Ensemble models** (Random Forest, XGBoost) outperform linear models in fraud detection tasks.
* The model can be extended into a **real-time fraud alert system** via API deployment.

---

## 🧾 Future Enhancements

* Implement **real-time detection** with FastAPI or Flask.
* Add **SHAP feature interpretability** for explainable AI.
* Integrate **streaming fraud alerts** with Kafka or AWS Lambda.
* Deploy a **dashboard** to monitor transactions and fraud risk levels.

---

## 📚 Tech Stack

* **Language:** Python 3.x
* **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, XGBoost
* **Environment:** Jupyter Notebook
* **Version Control:** Git & GitHub

---

## 🙌 Acknowledgments

Dataset inspired by simulated mobile money transaction data (e.g., [Kaggle PaySim Dataset](https://www.kaggle.com/datasets)).
Developed as part of a **Capstone Project** for applied machine learning in fraud analytics.

---

**Author:*** ASHOKA K U*
**License:** MIT License
