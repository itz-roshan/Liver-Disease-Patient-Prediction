<div align="center">

# 🩺 Liver Disease Patient Prediction

### An Artificial Neural Network (ANN) that predicts liver disease from routine blood-test parameters

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-brightgreen?style=for-the-badge)](#-license)

**[🚀 Live Demo](#-live-demo)** · **[📸 Preview](#-preview)** · **[⚙️ Installation](#%EF%B8%8F-installation--run-locally)** · **[🧠 Model](#-model-architecture)** · **[📊 Dataset](#-dataset)**

</div>

---

## 📌 Overview

**Liver Disease Patient Prediction** is an end-to-end Machine Learning project that uses a custom-built **Artificial Neural Network (ANN)**, trained with **Keras/TensorFlow**, to predict whether a patient is likely to have liver disease based on **10 clinical and blood-test features** — age, gender, bilirubin levels, liver enzymes, proteins, albumin, and the albumin/globulin ratio.

The project covers the **complete ML lifecycle**:

- 🧹 Data cleaning & preprocessing (missing value imputation, encoding)
- 🏗️ Building and training a feed-forward ANN with Keras
- 📈 Model evaluation
- 💾 Model persistence with `joblib`
- 🌐 An interactive **Streamlit** web app to serve real-time predictions in the browser

---

## 🚀 Live Demo

> ### 👉 **[Click here to try the live app](https://your-app-name.streamlit.app)** 👈

*(Replace the link above with your own Streamlit Community Cloud URL once deployed — see [Deployment](#-deploy-your-own-live-link) below. The app runs entirely in-browser and opens directly in Chrome or any modern browser.)*

---

## 📸 Preview

| Input Form | Prediction Result |
|:---:|:---:|
| Enter patient clinical parameters | Instant ANN-powered prediction with confidence score |

*(Add your own screenshots/GIF here — e.g. `assets/demo.gif` — once you run the app locally.)*

---

## ✨ Features

- 🎨 Clean, modern, responsive **Streamlit UI** with custom styling
- 🧠 Real ANN inference (Keras `Sequential` model, not a mock)
- 📊 Instant risk score, prediction label & confidence meter
- 🔍 Transparent — view the exact feature vector sent to the model
- ⚡ Cached model loading for fast repeated predictions
- 📱 Fully responsive layout — works on desktop and mobile browsers
- 🛡️ Input validation to prevent invalid clinical values

---

## 🧠 Model Architecture

Built with `keras.models.Sequential`:

```
Input Layer          : 10 features
Dense Layer 1         : 10 units, activation = ReLU
Dense Layer 2         : 6  units, activation = ReLU
Dense Layer 3         : 4  units, activation = ReLU
Output Layer          : 1  unit,  activation = Linear
```

| Setting | Value |
|---|---|
| Loss function | `mean_squared_error` |
| Optimizer | `adam` |
| Epochs | 50 |
| Batch size | 32 |
| Validation split | 10% |
| Train / Test split | 70% / 30% |
| Total trainable params | 209 |

The trained model is serialized to **`model.pkl`** using `joblib` and loaded directly by the Streamlit app for inference.

---

## 📊 Dataset

This project uses the **Indian Liver Patient Dataset (ILPD)**, sourced from the UCI Machine Learning Repository, containing **583 patient records** and the following features:

| # | Feature | Description |
|---|---|---|
| 1 | `Age` | Age of the patient |
| 2 | `Gender` | Male / Female |
| 3 | `Total_Bilirubin` | Total bilirubin level (mg/dL) |
| 4 | `Direct_Bilirubin` | Direct bilirubin level (mg/dL) |
| 5 | `Alkaline_Phosphotase` | ALP enzyme level (IU/L) |
| 6 | `Alamine_Aminotransferase` | SGPT enzyme level (IU/L) |
| 7 | `Aspartate_Aminotransferase` | SGOT enzyme level (IU/L) |
| 8 | `Total_Protiens` | Total protein level (g/dL) |
| 9 | `Albumin` | Albumin level (g/dL) |
| 10 | `Albumin_and_Globulin_Ratio` | A/G ratio |
| target | `Dataset` | 1 = Liver disease patient, 0 = Not a liver disease patient |

**Preprocessing steps applied:**
- Filled missing `Albumin_and_Globulin_Ratio` values with the column mean
- Encoded `Gender` (`Male → 0`, `Female → 1`)
- Remapped target labels (`2 → 0`) for binary classification

---

## 🗂️ Project Structure

```
liver-disease-prediction/
├── data/
│   └── indian_liver_patient.csv      # Raw dataset
├── Liver Disease Patient Prediction.ipynb   # EDA + model training notebook
├── model.pkl                          # Trained ANN model (Keras, saved via joblib)
├── app.py                             # Streamlit web app (UI + inference)
└── README.md                          # Project documentation
```

 
## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | TensorFlow / Keras |
| Data Handling | Pandas, NumPy |
| Preprocessing / Splitting | scikit-learn |
| Model Serialization | Joblib |
| Web App / UI | Streamlit |
| Notebook | Jupyter |

---

## 🔮 Future Improvements

- [ ] Add a sigmoid output layer + binary cross-entropy for true probabilistic classification
- [ ] Feature scaling (`StandardScaler`) integrated into the inference pipeline
- [ ] Model performance dashboard (confusion matrix, ROC-AUC, precision/recall)
- [ ] Hyperparameter tuning (dropout, learning rate, layer sizes)
- [ ] Batch prediction via CSV upload
- [ ] Docker support for one-command deployment

---

## ⚠️ Disclaimer

This project is built **for educational and portfolio purposes only**. Predictions generated by this app are **not** medical advice and should never be used as a substitute for professional diagnosis by a licensed healthcare provider.

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and share.

---

## 🙌 Acknowledgements

- Dataset: [Indian Liver Patient Dataset (ILPD) — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset)
- Built with [Streamlit](https://streamlit.io/) and [TensorFlow/Keras](https://www.tensorflow.org/)

<div align="center">

**⭐ If you found this project useful, consider giving it a star on GitHub! ⭐**

</div>
