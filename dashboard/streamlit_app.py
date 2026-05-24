import streamlit as st
import requests
import pandas as pd
import plotly.express as px

API_URL = "http://localhost:8000/predict"
st.set_page_config(page_title="Genre Classification Dashboard",layout="wide")
st.title("Music Genre Classification Dashboard")
uploaded = st.file_uploader("Upload WAV or MP3", type=["wav", "mp3"])

if uploaded:
    st.audio(uploaded)
    with st.spinner("Predicting..."):
        response = requests.post(
            API_URL,
            files={"file": uploaded}
        )

    results = response.json()
    rows = [[],[],[]]
    for model_name, result in results.items():
        if model_name.endswith("svm"):
            rows[0].append({
                "Model": model_name,
                "Prediction": result["prediction"],
                "Confidence": result["confidence"]
            })
        elif model_name.endswith("knn"):
            rows[1].append({
                "Model": model_name,
                "Prediction": result["prediction"],
                "Confidence": result["confidence"]
            })
        elif model_name.endswith("mlp"):
            rows[2].append({
                "Model": model_name,
                "Prediction": result["prediction"],
                "Confidence": result["confidence"]
            })

    df1 = pd.DataFrame(rows[0])
    st.subheader("SVM predictions")
    st.dataframe(df1)
    df2 = pd.DataFrame(rows[1])
    st.subheader("KNN predictions")
    st.dataframe(df2)
    df3 = pd.DataFrame(rows[2])
    st.subheader("MLP predictions")
    st.dataframe(df3)
    df = pd.concat([df1, df2, df3], ignore_index=True)
    fig = px.bar(
        df,
        x="Model",
        y="Confidence",
        color="Prediction",
        title="Confidence Comparison"
    )
    st.plotly_chart(fig, width='stretch')
    prediction_counts = (
        df["Prediction"]
        .value_counts()
        .reset_index()
    )

    prediction_counts.columns = ["Genre","Votes"]

    pie = px.pie(
        prediction_counts,
        values="Votes",
        names="Genre",
        title="Prediction Agreement"
    )

    st.plotly_chart(pie, width='stretch')