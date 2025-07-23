import streamlit as st
import pandas as pd
import pickle

# Define the models to load
model_files = {
    "Logistic Regression": "logreg_rfe_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "AdaBoost": "adaboost_model.pkl",
    "Gradient Boosting": "gb_model.pkl",
    "XGBoost": "xgb_model.pkl",
    "Decision Tree (CART)": "cart_model.pkl",
    "K-Nearest Neighbors (KNN)": "knn_model.pkl",
    "Naive Bayes": "nb_model.pkl"
}

# Features expected
feature_names = ['GRE Score', 'University Rating', 'CGPA']

# For statsmodels only
statsmodels_feature_names = ['const'] + feature_names

# App UI
st.title("📘 Admission Prediction App (Multi-Model)")
st.markdown("Choose a model and provide your input to predict admission outcome.")

# Model selection
selected_model_name = st.selectbox("Select Model", list(model_files.keys()))

# Load the selected model
try:
    with open(model_files[selected_model_name], "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# Inputs
gre = st.number_input("GRE Score", min_value=200, max_value=340, value=320)
rating = st.selectbox("University Rating", [1, 2, 3, 4, 5], index=3)
cgpa = st.slider("CGPA", min_value=0.0, max_value=10.0, step=0.1, value=8.5)

# Prediction logic
def prepare_input(gre, rating, cgpa, model_name):
    data = [[gre, rating, cgpa]]
    if model_name == "Logistic Regression":
        df = pd.DataFrame([[1.0, gre, rating, cgpa]], columns=statsmodels_feature_names)
    else:
        df = pd.DataFrame(data, columns=feature_names)
    return df

# Predict button
if st.button("Predict Admission"):
    input_df = prepare_input(gre, rating, cgpa, selected_model_name)
    
    # Prediction
    if selected_model_name == "Logistic Regression":
        prob = model.predict(input_df)[0]
    else:
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_df)[0][1]
        else:
            prob = model.predict(input_df)[0]  # fallback if model does not support probability
            st.warning("This model does not output probability. Interpreting raw class prediction.")

    label = "Admit" if prob >= 0.6 else "Reject"
    
    st.subheader(f"🎯 Result: **{label}**")
    st.write(f"📊 Probability of Admission: **{prob:.4f}**" if isinstance(prob, float) else f"Prediction: {prob}")

    if label == "Admit":
        st.success("Congratulations! Based on the input, admission is likely.")
    else:
        st.error("Unfortunately, admission is unlikely based on the input.")
