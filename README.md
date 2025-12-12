Here is your **final polished README.md** — rewritten cleanly, professionally, and formatted correctly for GitHub.

I have:

✔ Kept all information you provided
✔ Fixed structure, grammar, formatting
✔ Added missing setup instructions
✔ Updated repository link formatting
✔ Ensured it matches submission requirements

You can **copy–paste this entire block** directly into `README.md`.

---

# **Optimize Manufacturing Operations with a Predictive Maintenance Model**

### **Author:** *Kandula Vinay Gupta*

*Aditya College of Engineering and Technology*

---

## 📌 **Project Overview**

This project builds an end-to-end **Predictive Maintenance System** using industrial sensor time-series data from the **NASA Turbofan Engine (FD001)** dataset.

The goal is to shift from **reactive/scheduled maintenance** to **condition-based maintenance**, enabling organizations to:

* ⚙️ Reduce machine downtime
* 💸 Lower maintenance costs
* 📉 Prevent unexpected production failures

The model predicts whether an engine is **at risk of failure** within the next few cycles.

---

## 🚀 **Key Features**

| Component                       | Status       | Output                          |
| ------------------------------- | ------------ | ------------------------------- |
| Data cleaning & preprocessing   | ✔️ Completed | Final processed dataset         |
| Time-series feature engineering | ✔️ Completed | Rolling stats & lag features    |
| Time-aware ML modeling          | ✔️ Completed | RandomForest Classifier         |
| Data leakage prevention         | ✔️ Verified  | Sorted per-unit transformations |
| Model interpretability          | ✔️ Completed | SHAP visualizations             |
| Interactive Dashboard           | ✔️ Completed | Streamlit-based UI              |
| Evaluation & Reporting          | ✔️ Completed | Executive Summary PDF           |

---

## 🧠 **Machine Learning Workflow**

### **1️⃣ Exploratory Data Analysis (EDA)**

* Checked missing values
* Distribution of sensors analyzed
* Correlation across sensors
* RUL behavior inspected

### **2️⃣ Feature Engineering (Leakage-Free)**

* Rolling mean/min/max for windows: 5, 10, 20
* Gradient-based rate-of-change features
* Per-unit sorted, ensuring **no future values leak into past**

**Total engineered features:** *173*

### **3️⃣ Validation Strategy**

To prevent leakage:

✔ Used `TimeSeriesSplit`
✔ No shuffling
✔ Test sequences always follow training sequences

---

## 📊 **Model Performance**

| Metric                     | Score     |
| -------------------------- | --------- |
| Mean CV F1-Score (Class 1) | **0.845** |
| Holdout Test F1-Score      | **0.843** |

### 📌 **Confusion Matrix**

```
               Predicted
            0 (Healthy) | 1 (Failure)
-------------------------------------
True 0     |   3495     |    43
True 1     |   128      |   461
```

➡ The model meets project requirements: **F1 ≥ 0.75** for the minority class.

---

## 🔍 **Interpretability Insights (SHAP)**

Top predictors of failure include:

* Rolling STD of **sensor_3**
* Rolling mean of **sensor_7**
* Degradation patterns in **sensor_11**
* Long-term variations in **sensor_15**, **sensor_21**

SHAP helps maintenance engineers understand **why** a failure alert was triggered.

---

## 🖥️ **Streamlit Dashboard**

### ✔ Features:

* Select engine unit
* View latest operational cycle
* Predict health (Healthy / At Risk)
* Adjust model threshold
* Show feature importances & SHAP explanations

### ▶ Launch dashboard:

```bash
streamlit run dashboard/app.py
```

---

## 📁 **Repository Structure**

```
predictive_maintenance_project/
│
├── data/
│   ├── raw/                       
│   │   ├── train_FD001.txt
│   │   ├── test_FD001.txt
│   │   └── RUL_FD001.txt         
│   └── processed/                 
│       └── train_features_FD001_no_leak.csv
│
├── notebooks/                    
│   ├── 01_EDA.ipynb             
│   ├── 02_Feature_Engineering.ipynb 
│   ├── 03_Model_Training.ipynb    
│   └── 04_Model_Explainability_SHAP.ipynb 
├── dashboard/                    
│   └── app.py                     # Streamlit/Dash application file
│
├── models/                        # Trained machine learning model artifacts
│   ├── rf_FD001.joblib            # Serialized Random Forest (RF) model for FD001
│   └── rf_FD001_features.json     # List of features used to train the RF model
│
├── requirements.txt               # List of all Python packages and dependencies
├── README.md                      # Project description, installation, and usage instructions
├── executive_summary.pdf          # High-level overview document for stakeholders
└── video_demonstration.txt        # Text file containing link to the large video asset (e.g., Google Drive link)
```

---

## 🛠️ **Installation & Environment Setup**

### **Clone repository**

```bash
git clone https://github.com/vinay-gupta-kandula/Predictive-Maintenance-Model.git
cd predictive_maintenance_project
```

### **Create virtual environment**

Windows:

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

Mac/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## ▶ **Reproducibility Workflow**

| Step | File                                 | Output                     |
| ---- | ------------------------------------ | -------------------------- |
| 1    | `01_EDA.ipynb`                       | Explore dataset            |
| 2    | `02_Feature_Engineering.ipynb`       | Generate processed dataset |
| 3    | `03_Model_Training.ipynb`            | Train model & evaluation   |
| 4    | `04_Model_Explainability_SHAP.ipynb` | SHAP plots                 |
| 5    | `streamlit run dashboard/app.py`     | Dashboard UI               |

---

## 💼 **Business Impact**

* Prevents unplanned shutdowns
* Extends engine operational life
* Reduces repair & downtime costs
* Supports data-driven maintenance planning

---

## 🔮 **Future Enhancements**

* Add Remaining Useful Life (RUL) prediction
* Deploy dashboard on cloud (AWS / Azure)
* Add automated retraining pipeline
* Integrate cost optimization analytics

---

## 🏁 **Conclusion**

This project delivers a complete **Predictive Maintenance Solution** featuring:

✔ High-performance ML model
✔ Zero leakage feature engineering
✔ Explainable predictions
✔ Real-time dashboard
✔ Clean, reproducible documentation

---

## 👤 **Prepared by**

**Kandula Vinay Gupta**
Aditya College of Engineering and Technology
📧 *[kvinaygupta4242@gmail.com](mailto:kvinaygupta4242@gmail.com)*

---
