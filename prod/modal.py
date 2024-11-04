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
st.title("🤖 Explore Machine Learning Models")
st.markdown("""
    This application allows you to upload a dataset, explore different machine learning models, and compare their performances.
""")

# Step 1: Upload Dataset
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(data.head())

    # Select target and features
    target_column = st.selectbox("Select the target column:", options=data.columns)
    feature_columns = st.multiselect("Select feature columns:", options=[col for col in data.columns if col != target_column])

    if target_column and feature_columns:
        X = data[feature_columns]
        y = data[target_column]

        # Initialize an empty dictionary to store accuracies
        accuracies = {}

        # Define model information
        models_info = {
            "Logistic Regression": {
                "model": LogisticRegression(max_iter=200),
                "advantages": "Simple and efficient for binary classification.",
                "disadvantages": "Not suitable for non-linear data.",
                "image": "prod/assets/log.gif",
                "suitable_for": "Binary classification problems with a linear decision boundary.",
            },
            "Decision Tree": {
                "model": DecisionTreeClassifier(),
                "advantages": "Easy to interpret and visualize.",
                "disadvantages": "Prone to overfitting.",
                "image": "prod/assets/dec.gif",
                "suitable_for": "Classification and regression tasks with a clear decision structure.",
            },
            "Random Forest": {
                "model": RandomForestClassifier(),
                "advantages": "Reduces overfitting and improves accuracy.",
                "disadvantages": "Can be slow and resource-intensive.",
                "image": "prod/assets/rand.gif",
                "suitable_for": "Problems with high dimensionality and where overfitting is a concern.",
            },
            "Support Vector Machine": {
                "model": SVC(),
                "advantages": "Effective in high-dimensional spaces.",
                "disadvantages": "Less effective on very large datasets.",
                "image": "prod/assets/svm.gif",
                "suitable_for": "Classification tasks where the number of features is greater than the number of samples.",
            },
            "K-Nearest Neighbors": {
                "model": KNeighborsClassifier(),
                "advantages": "Simple and effective for small datasets.",
                "disadvantages": "Slow with large datasets.",
                "image": "prod/assets/knn.gif",
                "suitable_for": "Problems where the data is uniformly distributed.",
            },
            "Naive Bayes": {
                "model": GaussianNB(),
                "advantages": "Fast and works well with large datasets.",
                "disadvantages": "Assumes feature independence.",
                "image": "prod/assets/nb.gif",
                "suitable_for": "Text classification tasks and when the data is relatively independent.",
            },
            "Gradient Boosting": {
                "model": GradientBoostingClassifier(),
                "advantages": "Effective for large datasets and often achieves high accuracy.",
                "disadvantages": "Can be sensitive to noisy data.",
                "image": "prod/assets/gb.gif",
                "suitable_for": "Classification and regression tasks.",
            },
            "XGBoost": {
                "model": XGBClassifier(),
                "advantages": "Very efficient and can handle sparse data well.",
                "disadvantages": "Complexity can lead to overfitting.",
                "image": "prod/assets/xg.gif",
                "suitable_for": "Various structured data problems.",
            },
            "LightGBM": {
                "model": lgb.LGBMClassifier(),
                "advantages": "Fast training speed and high efficiency.",
                "disadvantages": "Can be less accurate on smaller datasets compared to other algorithms.",
                "image": "prod/assets/lgt.gif",
                "suitable_for": "Large datasets with a high number of features.",
            },
        }

        # Step 2: Model Explanation and Training
        for model_name, model_info in models_info.items():
            col1, col2 = st.columns([1, 2])

            with col1:
                st.image(model_info["image"], use_column_width=True)
            with col2:
                st.subheader(model_name)
                st.write(f"**Advantages:** {model_info['advantages']}")
                st.write(f"**Disadvantages:** {model_info['disadvantages']}")
                st.write(f"**Suitable For:** {model_info['suitable_for']}")

                if st.button(f"Train {model_name}"):
                    # Split the data
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model = model_info["model"]
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    accuracy = accuracy_score(y_test, predictions)
                    st.success(f"{model_name} Accuracy: {accuracy:.2f}")
                    accuracies[model_name] = accuracy
else:
    st.info("Please upload a dataset to proceed.")
