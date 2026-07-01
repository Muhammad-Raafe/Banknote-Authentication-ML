# 💵 Banknote Authentication AI

An interactive machine learning web app that detects whether a banknote is **genuine** or **forged**, using a K-Nearest Neighbors (KNN) classifier tuned with GridSearchCV — built with Scikit-learn and deployed with Streamlit.

🔗 **Live Demo:** https://banknote-appentication-ml-raafe.streamlit.app/

---

## 🧠 Overview

This project uses statistical features extracted from **wavelet-transformed images** of genuine and forged banknotes to classify currency authenticity. The KNN model is hyperparameter-tuned using `GridSearchCV` across multiple values of K, distance metrics, and weighting strategies, then wrapped in a premium, interactive dashboard for real-time predictions.

---

## ✨ Features

- 🔍 **Real-time note authentication** — input wavelet feature values and instantly get a Genuine/Forged prediction with confidence score
- ⚙️ **Hyperparameter tuning dashboard** — view the best K, distance metric, and weighting strategy found via GridSearchCV
- 📊 **Full model evaluation** — accuracy, cross-validation score, and confusion matrix
- 📈 **Interactive Plotly visualizations** — confidence bar chart, correlation heatmap, class distribution pie chart, and a customizable feature scatter plot
- 🎨 **Premium dark-themed UI** with gradient accents and smooth interactions
- ⚡ **Cached model training** for fast app performance

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| ML Library | Scikit-learn |
| Model | K-Nearest Neighbors (KNN) |
| Hyperparameter Tuning | GridSearchCV (5-fold CV) |
| Preprocessing | StandardScaler |
| Visualization | Plotly |
| Web Framework | Streamlit |
| Data Handling | Pandas, NumPy |

---

## 📂 Dataset

The model is trained on the **Banknote Authentication Dataset**, containing statistical features extracted from wavelet-transformed images of genuine and forged banknote-like specimens.

**Features:**
- `variance` — Variance of Wavelet Transformed image
- `skewness` — Skewness of Wavelet Transformed image
- `curtosis` — Curtosis of Wavelet Transformed image
- `entropy` — Entropy of image
- `class` — Target (0 = Genuine, 1 = Forged)

Dataset source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/267/banknote+authentication)

---

## ⚙️ How It Works

1. **Data Preprocessing** — Checked for nulls, inspected data types and distributions
2. **Feature/Target Split** — Separated wavelet features (`x`) from the authenticity label (`y`)
3. **Train/Test Split** — 80/20 split for training and evaluation
4. **Feature Scaling** — Applied `StandardScaler` (fit on train, transform on test only — no data leakage), essential since KNN is distance-based
5. **Hyperparameter Tuning** — Used `GridSearchCV` to find the optimal `n_neighbors`, `weights`, and `metric` across 5-fold cross-validation
6. **Evaluation** — Assessed using accuracy score, confusion matrix, and classification report
7. **Deployment** — Wrapped in an interactive Streamlit app for live predictions

---

## 🚀 Running Locally

1. Clone the repository
```bash
git clone https://github.com/Muhammad-Raafe/Banknote-Authentication-ML.git
cd Banknote-Authentication-ML
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the app
```bash
streamlit run app.py
```

Make sure `BankNote_Authentication.csv` is present in the project root directory.

---

## 📁 Project Structure

```
Banknote-Authentication-ML/
│
├── app.py                          # Streamlit web app
├── BankNote_Authentication.csv      # Dataset
├── requirements.txt                 # Python dependencies
└── README.md                         # Project documentation
```

---

## 📊 Model Performance

The tuned KNN model achieves high accuracy on the test set. Best hyperparameters (K value, distance metric, and weighting strategy) along with cross-validation score are displayed live in the app's sidebar.

---

## 🔮 Future Improvements

- Compare KNN performance against Logistic Regression, SVM, and Random Forest
- Add PCA-based 2D visualization of decision boundaries
- Support batch CSV upload for bulk note verification
- Add model explainability (e.g., feature importance via permutation)

---

## 👤 Author

**Muhammad Raafe**
AI/ML Enthusiast | Building a portfolio in Machine Learning & Data Science

GitHub: [@Muhammad-Raafe](https://github.com/Muhammad-Raafe)
