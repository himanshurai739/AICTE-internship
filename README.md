# 💧 Water Quality Prediction using Machine Learning

This project predicts whether water is **safe** or **unsafe** for use based on various water quality parameters using a trained machine learning model. The project includes data preprocessing, model training, saving the model, and deploying a Streamlit web app.

---

## 📌 Project Features

- ✅ Trained machine learning model (Random Forest)
- ✅ User-friendly Streamlit web app
- ✅ Input features: O2, Suspended solids, NH4, NO3, NO2, SO4, PO4, CL, BSK5
- ✅ Predicts: `Safe` or `Unsafe` water
- ✅ High accuracy with confusion matrix and classification report

---

## 📁 Files in this Repo

| File Name               | Description                                 |
|------------------------|---------------------------------------------|
| `PB_All_2000_2021.csv` | Original dataset with water quality records |
| `model_dev.py`         | Model training script                       |
| `water_quality_model.pkl` | Trained ML model saved with `pickle`     |
| `app.py`               | Streamlit app for real-time predictions     |
| `README.md`            | Project documentation                       |
| `requirements.txt`     | Required Python libraries                   |

---

## 📊 Dataset Description

The dataset contains historical water quality data from 2000
