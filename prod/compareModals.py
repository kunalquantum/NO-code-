import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tempfile



# Header Image
st.image("prod/assets/brain.gif", use_column_width=True)  # Replace with your image path

# Title and Introduction
st.title("🤖 Compare Machine Learning Models")
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

        # Define model information
        models_info = {
            "Logistic Regression": {
                "model": LogisticRegression(max_iter=200),
                "params": {'C': [0.01, 0.1, 1, 10]},
            },
            "Decision Tree": {
                "model": DecisionTreeClassifier(),
                "params": {'max_depth': [None, 5, 10, 20]},
            },
            "Random Forest": {
                "model": RandomForestClassifier(),
                "params": {'n_estimators': [10, 50, 100]},
            },
            "Support Vector Machine": {
                "model": SVC(),
                "params": {'C': [0.01, 0.1, 1, 10]},
            },
            "K-Nearest Neighbors": {
                "model": KNeighborsClassifier(),
                "params": {'n_neighbors': [3, 5, 7]},
            },
            "Naive Bayes": {
                "model": GaussianNB(),
                "params": {},
            },
        }

        # Step 3: Comparison of Models
        if st.button("Compare Models"):
            accuracies = {}  # Dictionary to store accuracies
            classification_reports = {}  # Dictionary to store classification reports

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            for model_name, model_info in models_info.items():
                model = model_info["model"]
                params = model_info["params"]

                # If there are parameters to tune, use GridSearchCV
                if params:
                    grid_search = GridSearchCV(model, params, cv=5)
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_
                    predictions = best_model.predict(X_test)
                else:
                    model.fit(X_train, y_train)  # Train model
                    predictions = model.predict(X_test)  # Make predictions

                accuracy = accuracy_score(y_test, predictions)  # Calculate accuracy
                accuracies[model_name] = accuracy  # Store accuracy
                classification_reports[model_name] = classification_report(y_test, predictions)  # Store classification report
                st.success(f"{model_name} trained with accuracy: {accuracy:.2f}")

            # Prepare data for plotting
            comparison_df = pd.DataFrame(list(accuracies.items()), columns=["Model", "Accuracy"])
            st.subheader("Comparison of Model Accuracies")
            st.dataframe(comparison_df)

            # Bar graph of accuracies
            plt.figure(figsize=(10, 5))
            sns.barplot(data=comparison_df, x="Model", y="Accuracy", palette="viridis")
            plt.title("Model Accuracies Comparison")
            plt.xticks(rotation=45)
            st.pyplot(plt)

            # Add a button to generate PDF report, keeping the state
            if st.button("Generate PDF Report"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
                    pdf_file_path = tmpfile.name

                # Generate PDF
                c = canvas.Canvas(pdf_file_path, pagesize=letter)
                c.drawString(100, 750, "Machine Learning Model Comparison Report")
                c.drawString(100, 720, f"Dataset: {uploaded_file.name}")
                c.drawString(100, 700, "Model Accuracies:")
                y_position = 680

                for model_name, accuracy in accuracies.items():
                    c.drawString(100, y_position, f"{model_name}: {accuracy:.2f}")
                    y_position -= 20

                c.drawString(100, y_position, "Classification Reports:")
                y_position -= 20

                for model_name, report in classification_reports.items():
                    c.drawString(100, y_position, f"{model_name} Report:")
                    y_position -= 20
                    for line in report.split('\n'):
                        c.drawString(120, y_position, line)
                        y_position -= 15
                    y_position -= 10  # Add some space between reports

                c.save()

                # Download PDF
                with open(pdf_file_path, "rb") as f:
                    st.download_button(
                        label="Download PDF Report",
                        data=f,
                        file_name="model_comparison_report.pdf",
                        mime="application/pdf"
                    )
else:
    st.info("Please upload a dataset to proceed.")
