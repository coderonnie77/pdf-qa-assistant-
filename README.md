# 🏠 House Price Predictor

An interactive, end-to-end Machine Learning web application that estimates residential real estate sale prices. Built using **Python**, **XGBoost**, and **Streamlit**, this project leverages advanced regression modeling and predictive analytics on the Ames Housing Dataset to deliver accurate, instantaneous home valuations.

🎯 **Live Application:** [View Live Dashboard](https://house-price-predictor.streamlit.app/)

---

## 🚀 Key Features
* **Dual-Model Evaluation:** Comprised of optimized Random Forest and Gradient Boosted Tree (XGBoost) pipelines.
* **Feature Engineering Pipeline:** Computes real-time analytical vectors including *House Age*, *Years Since Remodel*, and *Total Living Square Footage*.
* **Interpretability Layer:** Generates structural SHAP feature importance analysis tracking real-time local model weight decisions.
* **Interactive Dashboard:** Full-stack responsive web UI deploying intuitive numerical sliders, selections, and interactive data metrics cards.

---

## 📊 Model Performance Architecture

The predictive modeling engine was developed utilizing log-transformed scaling to stabilize variance and minimize skewness across raw financial targets. Out-of-sample stability is systematically verified using K-Fold cross-validation.

| Model Pipeline | Train $R^2$ | Cross-Validation $R^2$ | RMSE (Log Scale) |
| :--- | :---: | :---: | :---: |
| **Random Forest** | 0.864 | 0.843 | 0.1596 |
| **XGBoost (Selected)** | **0.876** | **0.869** | **0.1523** |

* **XGBoost Dollar Value Error (RMSE):** ~ $27,520 (Extremely low variance, indicating excellent out-of-sample data generalization without overfitting).

---

## 🛠️ Tech Stack & Architecture

* **Backend Modeling:** Python 3.13, Scikit-Learn, XGBoost, Pandas, NumPy, Joblib
* **Interpretability:** SHAP (SHapley Additive exPlanations)
* **Frontend UI Framework:** Streamlit
* **Visualizations:** Matplotlib
* **Deployment Environment:** Streamlit Community Cloud

---

## 📁 Repository Structure

```text
house-price-predictor/
├── data/
│   └── train.csv               # Raw housing dataset
├── model/
│   ├── train.py                # Model training and optimization pipeline
│   ├── xgb_model.pkl           # Core serialized production model artifact
│   ├── feature_names.pkl       # Serialized feature order checklist
│   └── shap_importance.png     # Static SHAP interpretation graph
├── notebooks/
│   └── explore.ipynb           # Exploratory Data Analysis (EDA)
├── app.py                      # Main Streamlit user interface entry point
├── requirements.txt            # Application package dependency definitions
└── README.md                   # Project documentation index
