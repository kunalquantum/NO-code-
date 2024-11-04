import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load a default dataset
default_data = sns.load_dataset("penguins")

# App title and description
st.title("🎉 Fun Data Exploration App!")
st.write("Upload a CSV file or use our sample penguins dataset to explore data step-by-step with expandable sections, fun images, and celebratory balloons 🎈.")

# Create three columns for the header image, centering it
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("prod/assets/datalogo.jfif", width=250)

# Step 1: Upload CSV file with expandable section
with st.expander("🔍 Step 1: Upload Your CSV File"):
    col1, col2 = st.columns([2, 1])
    
    with col1:
        dataset_choice = st.radio("Choose a dataset:", ("Upload your CSV file", "Use default Penguins dataset"))

        if dataset_choice == "Upload your CSV file":
            uploaded_file = st.file_uploader("Upload your CSV file here:", type='csv')
            if uploaded_file is not None:
                data = pd.read_csv(uploaded_file)
                st.write("Here's a preview of your data:")
                st.dataframe(data.head())
                st.balloons()  # Celebrate file upload success
        else:
            data = default_data
            st.write("Using the default Penguins dataset:")
            st.dataframe(data.head())
            st.balloons()

    with col2:
        if dataset_choice == "Upload your CSV file":
            st.image("prod/assets/peng.jpg", use_column_width=True)
        else:
            st.image("prod/assets/peng.jpg", use_column_width=True)
        
        # Button to access resources for Step 2
        if st.button("Resources (click here)", key='step1_resources'):
            st.markdown('<a href="https://www.kaggle.com/code/parulpandey/penguin-dataset-the-new-iris" target="_blank">Click here to access resources</a>', unsafe_allow_html=True)

# Step 2: Show Data Types in Expandable Section
if 'data' in locals():
    with st.expander("📊 Step 2: Explore Data Types"):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            data_types = data.dtypes
            st.write("Data types in your dataset:")
            st.write(data_types)

            # Display simplified data type labels
            simplified_data_types = {
                'int64': 'Integer Numbers',
                'float64': 'Decimal Numbers',
                'object': 'Categorical',
                'bool': 'True/False',
                'datetime64[ns]': 'Date/Time'
            }
            simplified_labels = data.dtypes.map(simplified_data_types)
            data_type_counts = simplified_labels.value_counts()
            st.bar_chart(data_type_counts)

            st.balloons()  # Celebrate after viewing data types

        with col2:
            st.image("prod/assets/datatypes.png", use_column_width=True)
            # Button to access resources for Step 3
            if st.button("Resources (click here)", key='step2_resources'):
                st.markdown('<a href="https://en.wikipedia.org/wiki/Data_type" target="_blank">Click here to access resources</a>', unsafe_allow_html=True)

# Step 3: Display Summary Statistics with Balloons
    with st.expander("📈 Step 3: View Summary Statistics"):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("Summary of Numerical Data")
            st.write(data.describe())
            st.balloons()  # Celebrate summary stats display

        with col2:
            st.image("prod/assets/summery.jpg", use_column_width=True)
            # Button to access resources for Step 4
            if st.button("Resources (click here)", key='step3_resources'):
                st.markdown('<a href="https://www.kaggle.com/code/parulpandey/penguin-dataset-the-new-iris" target="_blank">Click here to access resources</a>', unsafe_allow_html=True)

# Step 4: Unique Values in Categorical Columns
    with st.expander("🔎 Step 4: Unique Values in Categorical Columns"):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            categorical_cols = data.select_dtypes(include=['object']).columns
            st.write("Unique values in each categorical column:")
            for col in categorical_cols:
                st.write(f"{col}: {data[col].nunique()} unique values")

            st.balloons()  # Celebrate unique values display

        with col2:
            st.image("prod/assets/uni.jpg", use_column_width=True)
            # Button to access resources for Step 5
            if st.button("Resources (click here)", key='step4_resources'):
                st.markdown('<a href="https://www.kaggle.com/code/parulpandey/penguin-dataset-the-new-iris" target="_blank">Click here to access resources</a>', unsafe_allow_html=True)

# Step 5: Visualization Options in Expandable Section
    with st.expander("📊 Step 5: Visualize Your Data"):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            visual_choice = st.selectbox("Choose a visualization type", ["Histogram", "Bar Chart", "Scatter Plot"])

            if visual_choice == "Histogram":
                num_col = st.selectbox("Select a numerical column for histogram:", data.select_dtypes(include=['float64', 'int']).columns)
                fig, ax = plt.subplots()
                sns.histplot(data[num_col], bins=30, kde=True, ax=ax)
                st.pyplot(fig)
                st.balloons()

            elif visual_choice == "Bar Chart":
                cat_col = st.selectbox("Select a categorical column for bar chart:", categorical_cols)
                fig, ax = plt.subplots()
                sns.countplot(x=data[cat_col], ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)
                st.balloons()

            elif visual_choice == "Scatter Plot":
                x_col = st.selectbox("X-axis (numerical):", data.select_dtypes(include=['float64', 'int']).columns)
                y_col = st.selectbox("Y-axis (numerical):", data.select_dtypes(include=['float64', 'int']).columns)
                if x_col != y_col:
                    fig, ax = plt.subplots()
                    sns.scatterplot(x=data[x_col], y=data[y_col], ax=ax)
                    st.pyplot(fig)
                    st.balloons()

        with col2:
            if visual_choice == "Histogram":
                st.image("prod/assets/histo.jpg", use_column_width=True)
            elif visual_choice == "Bar Chart":
                st.image("prod/assets/bar.jpg", use_column_width=True)
            elif visual_choice == "Scatter Plot":
                st.image("prod/assets/scatter.jpg", use_column_width=True)
            # Button to access resources for Step 6
            if st.button("Resources (click here)", key='step5_resources'):
                st.markdown('<a href="https://www.kaggle.com/code/parulpandey/penguin-dataset-the-new-iris" target="_blank">Click here to access resources</a>', unsafe_allow_html=True)

# Step 6: Filtering Data in Expandable Section
    with st.expander("🔍 Step 6: Filter Your Data"):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("Select filters for your data:")
            filter_column = st.selectbox("Choose a column to filter by:", data.columns)
            unique_values = data[filter_column].dropna().unique()
            selected_value = st.selectbox(f"Select a value for {filter_column}:", unique_values)
            filtered_data = data[data[filter_column] == selected_value]
            st.write("Filtered Data:")
            st.dataframe(filtered_data)

        with col2:
            st.image("prod/assets/filter.jpg", use_column_width=True)
            # Button to access resources for Step 7
            if st.button("Resources (click here)", key='step6_resources'):
                st.markdown('<a href="https://www.kaggle.com/code/parulpandey/penguin-dataset-the-new-iris" target="_blank">Click here to access resources</a>', unsafe_allow_html=True)

# Step 7: Download Filtered Data
    with st.expander("💾 Step 7: Download Your Filtered Data"):
        if 'filtered_data' in locals() and filtered_data is not None:
            csv = filtered_data.to_csv(index=False)
            st.download_button(label="Download Filtered Data", data=csv, file_name='filtered_data.csv', mime='text/csv')

