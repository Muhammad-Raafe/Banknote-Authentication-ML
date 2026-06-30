import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Spam Message Classifier",
    page_icon="📩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- DARK THEME CSS ----------------
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #00d4ff;
        text-align: center;
        margin-bottom: 0px;
    }
    .sub-title {
        text-align: center;
        color: #9aa0a6;
        margin-bottom: 30px;
    }
    .result-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 15px;
    }
    .spam-box {
        background-color: #2d0000;
        border: 2px solid #ff4b4b;
        color: #ff4b4b;
    }
    .ham-box {
        background-color: #002d0a;
        border: 2px solid #00ff7f;
        color: #00ff7f;
    }
    div[data-testid="stMetric"] {
        background-color: #1c1f26;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2a2e37;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- LOAD + TRAIN (cached) ----------------
@st.cache_resource
def load_and_train():
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df = df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], errors="ignore")
    df.columns = ["label", "message"]

    le = LabelEncoder()
    df["label_encoded"] = le.fit_transform(df["label"])

    x_text = df["message"]
    y = df["label_encoded"]

    cv = CountVectorizer()
    x = cv.fit_transform(x_text)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, random_state=42, test_size=0.2
    )

    model = MultinomialNB()
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)

    acc = accuracy_score(y_test, prediction)
    cm = confusion_matrix(y_test, prediction)
    report = classification_report(y_test, prediction, target_names=le.classes_, output_dict=True)

    return model, cv, le, df, acc, cm, report

model, cv, le, df, acc, cm, report = load_and_train()

# ---------------- HEADER ----------------
st.markdown('<p class="main-title">📩 Spam Message Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Naive Bayes based SMS Spam Detection System</p>', unsafe_allow_html=True)

# ---------------- SIDEBAR: MODEL INFO ----------------
with st.sidebar:
    st.header("📊 Model Performance")
    st.metric("Accuracy", f"{acc*100:.2f}%")
    st.metric("Total Messages", len(df))
    st.metric("Spam Messages", int((df['label'] == 'spam').sum()))
    st.metric("Ham Messages", int((df['label'] == 'ham').sum()))

    st.divider()
    st.subheader("Classification Report")
    report_df = pd.DataFrame(report).transpose().round(2)
    st.dataframe(report_df, use_container_width=True)

# ---------------- MAIN: PREDICTION ----------------
col1, col2 = st.columns([1.3, 1])

with col1:
    st.subheader("🔍 Test a Message")
    user_input = st.text_area(
        "Enter a message to classify:",
        height=150,
        placeholder="e.g. Congratulations! You've won a free prize, click here to claim..."
    )

    predict_btn = st.button("Classify Message", type="primary", use_container_width=True)

    if predict_btn:
        if user_input.strip() == "":
            st.warning("Please enter a message first.")
        else:
            vec_input = cv.transform([user_input])
            pred = model.predict(vec_input)[0]
            proba = model.predict_proba(vec_input)[0]
            label = le.inverse_transform([pred])[0]

            spam_prob = proba[list(le.classes_).index("spam")] * 100
            ham_prob = proba[list(le.classes_).index("ham")] * 100

            if label == "spam":
                st.markdown(f'<div class="result-box spam-box">🚨 SPAM ({spam_prob:.1f}% confidence)</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-box ham-box">✅ NOT SPAM ({ham_prob:.1f}% confidence)</div>', unsafe_allow_html=True)

            # Probability bar chart
            fig = go.Figure(go.Bar(
                x=[ham_prob, spam_prob],
                y=["Ham", "Spam"],
                orientation='h',
                marker_color=["#00ff7f", "#ff4b4b"],
                text=[f"{ham_prob:.1f}%", f"{spam_prob:.1f}%"],
                textposition="auto"
            ))
            fig.update_layout(
                template="plotly_dark",
                title="Prediction Confidence",
                xaxis_title="Probability (%)",
                height=250,
                margin=dict(l=10, r=10, t=40, b=10),
                plot_bgcolor="#0e1117",
                paper_bgcolor="#0e1117"
            )
            st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📈 Confusion Matrix")
    fig_cm = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=list(le.classes_),
        y=list(le.classes_)
    )
    fig_cm.update_layout(
        template="plotly_dark",
        height=350,
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117"
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    st.subheader("📊 Class Distribution")
    dist_df = df["label"].value_counts().reset_index()
    dist_df.columns = ["label", "count"]
    fig_dist = px.pie(
        dist_df, names="label", values="count",
        color="label",
        color_discrete_map={"ham": "#00ff7f", "spam": "#ff4b4b"},
        hole=0.4
    )
    fig_dist.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117"
    )
    st.plotly_chart(fig_dist, use_container_width=True)

# ---------------- SAMPLE DATA ----------------
with st.expander("📋 View Sample Dataset"):
    st.dataframe(df[["label", "message"]].sample(10, random_state=1), use_container_width=True)
