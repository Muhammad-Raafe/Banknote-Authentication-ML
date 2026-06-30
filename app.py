import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Banknote Authentication AI",
    page_icon="💵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- VIP DARK THEME CSS ----------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at top left, #141821 0%, #0a0c10 100%);
        color: #f5f6fa;
    }

    .main-title {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #00f5d4, #00bbf9, #9b5de5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        letter-spacing: 1px;
    }

    .sub-title {
        text-align: center;
        color: #8b92a8;
        font-size: 1.05rem;
        margin-bottom: 35px;
        font-weight: 400;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #161a23, #1d212c);
        padding: 18px;
        border-radius: 16px;
        border: 1px solid #262b38;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }

    div[data-testid="stMetric"] label {
        color: #8b92a8 !important;
    }

    .result-box {
        padding: 28px;
        border-radius: 18px;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 800;
        margin-top: 18px;
        letter-spacing: 0.5px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.4);
    }

    .real-box {
        background: linear-gradient(145deg, #003d2b, #002417);
        border: 2px solid #00f5a0;
        color: #00f5a0;
    }

    .fake-box {
        background: linear-gradient(145deg, #3d0014, #240009);
        border: 2px solid #ff4d6d;
        color: #ff4d6d;
    }

    .stButton>button {
        background: linear-gradient(90deg, #00bbf9, #9b5de5);
        color: white;
        font-weight: 700;
        border-radius: 12px;
        border: none;
        padding: 12px 0px;
        font-size: 1rem;
        transition: 0.3s;
    }

    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 20px rgba(0, 187, 249, 0.5);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #10131a, #0a0c10);
        border-right: 1px solid #1e222c;
    }

    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #00bbf9;
        margin-top: 10px;
        margin-bottom: 10px;
        border-left: 4px solid #9b5de5;
        padding-left: 10px;
    }

    hr {
        border-color: #1e222c;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- LOAD + TRAIN (cached) ----------------
@st.cache_resource
def load_and_train():
    df = pd.read_csv("BankNote_Authentication.csv")

    x = df.drop("class", axis=1)
    y = df["class"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = KNeighborsClassifier()
    param_grid = {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"]
    }

    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring="accuracy")
    grid.fit(x_train_scaled, y_train)
    prediction = grid.predict(x_test_scaled)

    acc = accuracy_score(y_test, prediction)
    cm = confusion_matrix(y_test, prediction)
    report = classification_report(y_test, prediction, output_dict=True)

    return df, scaler, grid, acc, cm, report, x.columns.tolist()

df, scaler, grid, acc, cm, report, feature_names = load_and_train()

# ---------------- HEADER ----------------
st.markdown('<p class="main-title">💵 Banknote Authentication AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">KNN-Powered Real vs Fake Currency Detection System</p>', unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown('<p class="section-header">⚙️ Model Insights</p>', unsafe_allow_html=True)
    st.metric("Model Accuracy", f"{acc*100:.2f}%")
    st.metric("Best K (n_neighbors)", grid.best_params_["n_neighbors"])
    st.metric("Best Weight Strategy", grid.best_params_["weights"].title())
    st.metric("Best Distance Metric", grid.best_params_["metric"].title())
    st.metric("Cross-Val Score", f"{grid.best_score_*100:.2f}%")

    st.divider()
    st.markdown('<p class="section-header">📋 Dataset Snapshot</p>', unsafe_allow_html=True)
    st.metric("Total Samples", len(df))
    st.metric("Genuine Notes", int((df["class"] == 0).sum()))
    st.metric("Forged Notes", int((df["class"] == 1).sum()))

# ---------------- MAIN: LIVE PREDICTION ----------------
st.markdown('<p class="section-header">🔍 Test a Banknote</p>', unsafe_allow_html=True)
st.caption("Enter the wavelet-transformed image features extracted from the banknote.")

col1, col2 = st.columns([1.3, 1])

with col1:
    c1, c2 = st.columns(2)
    with c1:
        variance = st.number_input("Variance of Wavelet", value=float(df["variance"].mean()), format="%.4f")
        skewness = st.number_input("Skewness of Wavelet", value=float(df["skewness"].mean()), format="%.4f")
    with c2:
        curtosis = st.number_input("Curtosis of Wavelet", value=float(df["curtosis"].mean()), format="%.4f")
        entropy = st.number_input("Entropy", value=float(df["entropy"].mean()), format="%.4f")

    predict_btn = st.button("🚀 Authenticate Note", use_container_width=True)

    if predict_btn:
        input_data = pd.DataFrame([[variance, skewness, curtosis, entropy]], columns=feature_names)
        input_scaled = scaler.transform(input_data)
        pred = grid.predict(input_scaled)[0]
        proba = grid.predict_proba(input_scaled)[0]

        real_prob = proba[0] * 100
        fake_prob = proba[1] * 100

        if pred == 0:
            st.markdown(f'<div class="result-box real-box">✅ GENUINE NOTE ({real_prob:.1f}% confidence)</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-box fake-box">🚨 FORGED NOTE ({fake_prob:.1f}% confidence)</div>', unsafe_allow_html=True)

        fig = go.Figure(go.Bar(
            x=[real_prob, fake_prob],
            y=["Genuine", "Forged"],
            orientation="h",
            marker_color=["#00f5a0", "#ff4d6d"],
            text=[f"{real_prob:.1f}%", f"{fake_prob:.1f}%"],
            textposition="auto"
        ))
        fig.update_layout(
            template="plotly_dark",
            title="Prediction Confidence",
            xaxis_title="Probability (%)",
            height=250,
            margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor="#0a0c10",
            paper_bgcolor="#0a0c10"
        )
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown('<p class="section-header">📊 Confusion Matrix</p>', unsafe_allow_html=True)
    fig_cm = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="Tealgrn",
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["Genuine", "Forged"],
        y=["Genuine", "Forged"]
    )
    fig_cm.update_layout(
        template="plotly_dark",
        height=320,
        margin=dict(l=10, r=10, t=20, b=10),
        plot_bgcolor="#0a0c10",
        paper_bgcolor="#0a0c10"
    )
    st.plotly_chart(fig_cm, use_container_width=True)

# ---------------- EXPLORATORY VISUALS ----------------
st.divider()
st.markdown('<p class="section-header">📈 Data Exploration</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔗 Feature Correlation", "📊 Class Distribution", "🌌 Feature Pairplot"])

with tab1:
    corr = df.corr()
    fig_corr = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r",
        aspect="auto"
    )
    fig_corr.update_layout(
        template="plotly_dark", height=450,
        plot_bgcolor="#0a0c10", paper_bgcolor="#0a0c10"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

with tab2:
    dist_df = df["class"].map({0: "Genuine", 1: "Forged"}).value_counts().reset_index()
    dist_df.columns = ["Class", "Count"]
    fig_dist = px.pie(
        dist_df, names="Class", values="Count", hole=0.45,
        color="Class",
        color_discrete_map={"Genuine": "#00f5a0", "Forged": "#ff4d6d"}
    )
    fig_dist.update_layout(
        template="plotly_dark", height=400,
        plot_bgcolor="#0a0c10", paper_bgcolor="#0a0c10"
    )
    st.plotly_chart(fig_dist, use_container_width=True)

with tab3:
    feature_x = st.selectbox("X-axis feature", feature_names, index=0)
    feature_y = st.selectbox("Y-axis feature", feature_names, index=1)
    plot_df = df.copy()
    plot_df["class"] = plot_df["class"].map({0: "Genuine", 1: "Forged"})
    fig_scatter = px.scatter(
        plot_df, x=feature_x, y=feature_y, color="class",
        color_discrete_map={"Genuine": "#00f5a0", "Forged": "#ff4d6d"},
        opacity=0.75
    )
    fig_scatter.update_layout(
        template="plotly_dark", height=450,
        plot_bgcolor="#0a0c10", paper_bgcolor="#0a0c10"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------- RAW DATA ----------------
with st.expander("📋 View Raw Dataset Sample"):
    st.dataframe(df.sample(10, random_state=1), use_container_width=True)
