import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score



# Title and Introduction
st.title("🤖 Model Training and Testing App")
st.markdown("""
    This application allows you to upload a dataset, train different machine learning models,
    and test them within the same interface. Choose a model, train it on your data, and make predictions with ease!
""")

# Section 1: Model Training
st.header("🔧 Train a Machine Learning Model")

with st.expander("Upload Dataset and Select Target and Features"):
    uploaded_file = st.file_uploader("Upload your CSV file for training", type="csv")

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.write(data.head())

        # Select target and features
        target_column = st.selectbox("Select the target column:", options=data.columns)
        feature_columns = st.multiselect("Select feature columns:", options=[col for col in data.columns if col != target_column])

# Step 2: Model Selection and Training
if uploaded_file and target_column and feature_columns:
    X = data[feature_columns]
    y = data[target_column]

    with st.expander("Choose Model and Train"):
        model_options = {
            "Logistic Regression": LogisticRegression(max_iter=200),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "Support Vector Machine": SVC(),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
            "XGBoost": XGBClassifier(),
            "LightGBM": lgb.LGBMClassifier(),
        }

        # Select a model
        selected_model_name = st.selectbox("Select a model to train:", list(model_options.keys()))
        selected_model = model_options[selected_model_name]

        # Train the selected model
        if st.button("Train Model"):
            with st.spinner("Training the model..."):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                selected_model.fit(X_train, y_train)
                predictions = selected_model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)

                st.success(f"**{selected_model_name} Model Accuracy:** {accuracy:.2f}")
                st.session_state["trained_model"] = selected_model
                st.session_state["feature_columns"] = feature_columns
                st.session_state["target_column"] = target_column
                st.session_state["accuracy"] = accuracy

# Section 2: Model Testing
st.header("🧪 Test the Trained Model")

if "trained_model" in st.session_state:
    with st.expander("Provide Inputs for Prediction"):
        st.write(f"**Model Selected for Testing:** {selected_model_name}")
        st.write(f"**Model Accuracy on Test Set:** {st.session_state['accuracy']:.2f}")

        input_data = []
        for feature in st.session_state["feature_columns"]:
            input_val = st.number_input(f"Enter value for {feature}", value=0.0)
            input_data.append(input_val)

        # Make Prediction
        if st.button("Predict"):
            with st.spinner("Making prediction..."):
                input_df = pd.DataFrame([input_data], columns=st.session_state["feature_columns"])
                prediction = st.session_state["trained_model"].predict(input_df)
                st.write("### Prediction Result:")
                st.write(prediction[0])
else:
    st.info("Please train a model in the first section before testing.")
