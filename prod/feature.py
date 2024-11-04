import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


# Title and Introduction
st.title("📊 Interactive Feature Selection for Machine Learning")
st.markdown("""
    Feature selection helps in identifying the most important features in a dataset for building efficient models. 
    This page guides you through the feature selection process with:
    - Simple explanations of each technique.
    - Hands-on interactions for applying each method to your dataset.
""")

# Step 1: Upload Dataset
def display_step_with_image(step_number, title, image_path, description, resource_link):
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_path, use_column_width=True)  # Displaying image
    with col2:
        st.subheader(title)
        st.write(description)
        if st.button("Resources (click here)", key=f'resource_{step_number}'):
            st.markdown(f'<a href="{resource_link}" target="_blank">Click here to access resources</a>', unsafe_allow_html=True)

display_step_with_image(
    step_number=1,
    title="Step 1: Upload Your Dataset",
    image_path="prod/assets/datam.gif",  # Replace with your actual image path
    description="🔄 Upload a CSV file with labeled data (including the target column).",
    resource_link="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html"  # Resource link for reading CSV
)

uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(data.head())

    # Select target and features
    target_column = st.selectbox("Select the target column (outcome variable):", options=data.columns)
    feature_columns = st.multiselect("Select the features you want to analyze:", options=[col for col in data.columns if col != target_column])

    if target_column and feature_columns:
        X = data[feature_columns]
        y = data[target_column]

        # Step 2: Data Preprocessing
        display_step_with_image(
            step_number=2,
            title="Step 2: Data Preprocessing",
            image_path="prod/assets/missing.gif",  # Replace with your actual image path
            description="⚙️ Standardize numerical features for consistent scaling.",
            resource_link="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html"  # Resource link for StandardScaler
        )
        
        scale_features = st.checkbox("Apply Standardization", help="Scaling is recommended for models sensitive to feature scales.")
        if scale_features:
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=feature_columns)
            st.write("Standardized Features:")
            st.dataframe(X.head())

        # Step 3: Feature Selection Techniques
        display_step_with_image(
            step_number=3,
            title="Step 3: Feature Selection Techniques",
            image_path="prod/assets/feature.gif",  # Replace with your actual image path
            description="📈 Choose a method to evaluate the importance of your features.",
            resource_link="https://scikit-learn.org/stable/modules/feature_selection.html"  # Resource link for feature selection
        )
        
        # Information-based methods
        st.subheader("Information-Based Methods")
        st.write("Information-based methods assess the relationships between features and the target. Select a method to view the most relevant features:")
        method = st.radio("Choose a method:", options=["Chi-Square", "Mutual Information"])

        selected_features = []
        
        if method == "Chi-Square":
            st.write("Chi-Square test evaluates the independence of features with respect to the target.")
            k = st.slider("Select number of top features to display:", min_value=1, max_value=len(feature_columns))
            chi2_selector = SelectKBest(chi2, k=k)
            X_chi2 = chi2_selector.fit_transform(X, y)
            chi2_features = [feature_columns[i] for i in chi2_selector.get_support(indices=True)]
            st.write(f"Top {k} features selected by Chi-Square test:", chi2_features)
            selected_features.extend(chi2_features)

        elif method == "Mutual Information":
            st.write("Mutual Information measures the dependency between features and target.")
            k = st.slider("Select number of top features to display:", min_value=1, max_value=len(feature_columns))
            mutual_info = mutual_info_classif(X, y)
            mutual_info_series = pd.Series(mutual_info, index=feature_columns).sort_values(ascending=False)
            mi_features = mutual_info_series.head(k).index.tolist()
            st.write(f"Top {k} features selected by Mutual Information:", mi_features)
            selected_features.extend(mi_features)

        # Model-based methods
        st.subheader("Model-Based Methods")
        st.write("Model-based methods use a machine learning model to assess feature importance. Choose Random Forest for quick insights.")
        
        rf_importance = st.button("Compute Feature Importance with Random Forest")
        rf_features = []

        if rf_importance:
            rf = RandomForestClassifier()
            rf.fit(X, y)
            importances = pd.Series(rf.feature_importances_, index=feature_columns).sort_values(ascending=False)
            st.write("Feature Importance Scores:")
            st.bar_chart(importances)
            rf_features = importances.head(5).index.tolist()
            st.write("Most important features as identified by Random Forest:", rf_features)
            selected_features.extend(rf_features)

        # Step 4: Conclusion and Summary of Selected Features
        display_step_with_image(
            step_number=4,
            title="Step 4: Conclusion",
            image_path="prod/assets/conclusion.gif",  # Replace with your actual image path
            description="✅ Here are the most relevant features identified from the methods above.",
            resource_link="https://scikit-learn.org/stable/modules/feature_selection.html#feature-selection"  # Resource link for feature selection summary
        )
        
        # Get unique features from the list of selected features from all methods
        unique_features = list(set(selected_features))
        st.write("Most Relevant Features Across All Methods:", unique_features)

        # Download the reduced dataset with the selected features
        if st.button("Download Dataset with Selected Features"):
            reduced_data = data[unique_features + [target_column]]
            csv = reduced_data.to_csv(index=False)
            st.download_button(label="Download Selected Features CSV", data=csv, file_name='selected_features_data.csv', mime='text/csv')
else:
    st.info("Please upload a dataset to proceed.")
