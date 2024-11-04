import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from io import BytesIO 


# App title and description with styling
st.title("🧹 Data Cleaning and Preprocessing")
st.markdown("""
    Prepare your dataset for analysis with these easy-to-follow steps!
    Use the sections below to clean and preprocess your data efficiently.
""")

# Row 1: Step 1 - Upload Dataset
col1, col2 = st.columns(2)

with col1:
    st.image("prod/assets/datam.gif", use_column_width=True)  # Replace with your image URL
    st.button("Resources", key="upload_resource", on_click=lambda: st.write("Upload your dataset."))

with col2:
    st.subheader("Step 1: Upload Your Dataset")
    st.write("🔄 Upload your CSV file here:")
    uploaded_file = st.file_uploader("", type='csv')
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Here's a preview of your data:")
        st.dataframe(data.head(), height=200)
        st.balloons()  # Celebrate file upload success

# Check if data is uploaded
if 'data' in locals():
    st.markdown("---")  # Horizontal line

    # Row 2: Step 2 - Handle Missing Values
    col3, col4 = st.columns(2)

    with col3:
        st.image("prod/assets/missing.gif", use_column_width=True)  # Replace with your image URL
        st.button("Resources", key="missing_values_resource", on_click=lambda: st.write("Handling missing values."))

    with col4:
        st.subheader("Step 2: Handle Missing Values")
        with st.expander("More Info on Handling Missing Values"):
            st.write("""
                Missing values can cause issues in analysis. 
                You can either drop rows with missing values or fill them with statistical measures.
            """)

        st.write("Identify and handle missing values in your dataset:")
        st.write(data.isnull().sum())  # Show missing values

        # Option to fill or drop missing values
        fill_action = st.radio("Select action for missing values:", ("Drop rows", "Fill with mean/median/mode"), index=0)

        if fill_action == "Drop rows":
            data = data.dropna()
            st.write("Rows with missing values have been dropped.")
        else:
            fill_strategy = st.selectbox("Select fill strategy:", ["Mean", "Median", "Mode"])
            for col in data.columns:
                if data[col].dtype in [np.float64, np.int64]:
                    imputer = SimpleImputer(strategy=fill_strategy.lower())
                    data[col] = imputer.fit_transform(data[[col]])
                else:
                    data[col].fillna(data[col].mode()[0], inplace=True)
            st.write("Missing values have been filled.")

        st.dataframe(data.head(), height=200)  # Show updated data
        st.balloons()

    st.markdown("---")  # Horizontal line

    # Row 3: Step 3 - Encode Categorical Variables
    col5, col6 = st.columns(2)

    with col5:
        st.image("prod/assets/category.gif", use_column_width=True)  # Replace with your image URL
        st.button("Resources", key="encoding_resource", on_click=lambda: st.write("Encoding categorical variables."))

    with col6:
        st.subheader("Step 3: Encode Categorical Variables")
        with st.expander("More Info on Encoding Categorical Variables"):
            st.write("""
                Categorical variables need to be converted into numerical form for analysis. 
                We can use Label Encoding or One-Hot Encoding based on the nature of the variable.
            """)

        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        st.write("Select the categorical columns to encode:")
        selected_cols = st.multiselect("Categorical columns:", categorical_cols)

        if selected_cols:
            encoding_method = st.radio("Select encoding method:", ("Label Encoding", "One-Hot Encoding"), index=0)
            for col in selected_cols:
                if encoding_method == "Label Encoding":
                    le = LabelEncoder()
                    data[col] = le.fit_transform(data[col])
                else:  # One-Hot Encoding
                    data = pd.get_dummies(data, columns=[col], drop_first=True)
            st.write("Categorical columns have been encoded.")
            st.dataframe(data.head(), height=200)  # Show updated data
            st.balloons()

    st.markdown("---")  # Horizontal line

    # Row 4: Step 4 - Scale Numerical Features
    col7, col8 = st.columns(2)

    with col7:
        st.image("prod/assets/scale.gif", use_column_width=True)  # Replace with your image URL
        st.button("Resources", key="scaling_resource", on_click=lambda: st.write("Scaling numerical features."))

    with col8:
        st.subheader("Step 4: Scale Numerical Features")
        with st.expander("More Info on Scaling Numerical Features"):
            st.write("""
                Scaling is essential to bring numerical values onto a similar scale, 
                which helps in improving model performance and convergence speed.
            """)

        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        st.write("Select numerical columns to scale:")
        selected_numeric_cols = st.multiselect("Numerical columns:", numeric_cols)

        if selected_numeric_cols:
            scaler = StandardScaler()
            data[selected_numeric_cols] = scaler.fit_transform(data[selected_numeric_cols])
            st.write("Numerical columns have been scaled.")
            st.dataframe(data.head(), height=200)  # Show updated data
            st.balloons()

    st.markdown("---")  # Horizontal line

    # Row 5: Step 5 - Visualize Cleaned Data
    col9, col10 = st.columns(2)

    with col9:
        st.image("prod/assets/visualize.gif", use_column_width=True)  # Replace with your image URL
        st.button("Resources", key="visualization_resource", on_click=lambda: st.write("Visualizing cleaned data."))

    with col10:
        st.subheader("Step 5: Visualize Cleaned Data")
        with st.expander("More Info on Visualizing Cleaned Data"):
            st.write("""
                Visualizing your data can help understand the distribution and 
                relationships between features after preprocessing.
            """)

        visual_choice = st.selectbox("Choose a visualization type", ["Histogram", "Box Plot", "Scatter Plot"])

        if visual_choice == "Histogram":
            col_name = st.selectbox("Select a numerical column for histogram:", numeric_cols)
            fig, ax = plt.subplots()
            sns.histplot(data[col_name], bins=30, kde=True, ax=ax)
            st.pyplot(fig)
            st.balloons()

        elif visual_choice == "Box Plot":
            col_name = st.selectbox("Select a numerical column for box plot:", numeric_cols)
            fig, ax = plt.subplots()
            sns.boxplot(x=data[col_name], ax=ax)
            st.pyplot(fig)
            st.balloons()

        elif visual_choice == "Scatter Plot":
            x_col = st.selectbox("Select x-axis column:", numeric_cols)
            y_col = st.selectbox("Select y-axis column:", numeric_cols)
            fig, ax = plt.subplots()
            sns.scatterplot(x=data[x_col], y=data[y_col], ax=ax)
            st.pyplot(fig)
            st.balloons()

    st.markdown("---")  # Horizontal line

    # Row 6: Step 6 - Download Cleaned Data
    col11, col12 = st.columns(2)

    with col11:
        st.image("prod/assets/download.gif", use_column_width=True)  # Replace with your image URL
        st.button("Resources", key="download_resource", on_click=lambda: st.write("Download your cleaned data."))

    
with col12:
    st.subheader("Step 6: Download Your Cleaned Data")

    with st.expander("More Info on Downloading Cleaned Data"):
        st.write("""
            Once you have cleaned and processed your data, you can download 
            it in CSV or Excel format for further analysis or modeling.
        """)

    # CSV download
    csv = data.to_csv(index=False).encode('utf-8')  # Encode CSV for download

    st.download_button(
        label="Download Cleaned Data (CSV)",
        data=csv,
        file_name='cleaned_data.csv',
        mime='text/csv'
    )

    # Excel download
    excel_buffer = BytesIO()  # Create an in-memory buffer
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        data.to_excel(writer, index=False, sheet_name='Cleaned Data')

    excel_buffer.seek(0)  # Move the cursor to the beginning of the buffer

    st.download_button(
        label="Download Cleaned Data (Excel)",
        data=excel_buffer,
        file_name='cleaned_data.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
# Run the app
if __name__ == '__main__':
    st.write("Feel free to explore your data cleaning journey!")

