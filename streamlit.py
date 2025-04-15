import streamlit as st
import pandas as pd
import joblib  # or pickle depending on your model
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import seaborn as sns

st.set_page_config(layout="wide")

# Load model only once
@st.cache_resource
def load_model():
    return joblib.load("autism_model.pkl")  # Update with your model file

# Sidebar navigation
st.sidebar.title("Menu")
selection = st.sidebar.radio("Go to", ["About", "EDA", "Prediction"])

# Page: About
if selection == "About":
    st.title("🧠 Autism Detection using Machine Learning Approach")
    st.write("""
    This app helps explore patterns in data related to autism and uses a machine learning model to make predictions.
    
    - **EDA Page**: View visualizations and data insights  
    - **Prediction Page**: Input features to predict autism likelihood
    """)

# Page: EDA
elif selection == "EDA":
    st.title("📊 Exploratory Data Analysis")
    
    # Load your dataset
    df = pd.read_csv("adult.csv")  # Replace with your actual file

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # st.write("### Correlation Heatmap")
    # plt.figure(figsize=(10,6))
    # sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    # st.pyplot(plt)

    # st.write("### Class Distribution")
    # sns.countplot(data=df, x='Class')  # Update 'Class' with your target column
    # st.pyplot(plt)
    tableau_url = "https://public.tableau.com/views/Autismdistributionbygender/Dashboard1?:embed=true&:showVizHome=no"
    components.iframe(tableau_url, height=1100, width =1200 )
    

elif selection == "Prediction":
    st.title("🤖 Predict Autism")

    @st.cache_resource
    def load_models():
        model_adult = joblib.load("best_adult_model.joblib")
        model_child = joblib.load("best_children_model.joblib")
        return model_adult, model_child

    model_adult, model_child = load_models()

    # 👤 Demographics
    age = st.slider("Age", 1, 100, 30)
    gender = st.selectbox("Gender", ["male", "female"])
    ethnicity = st.selectbox("Ethnicity", ["White-European", "Black", "Asian", "Hispanic", "Latino", "South Asian", "Middle Eastern", "Pasifika", "Others", "Turkish", "Unknown"])
    country = st.selectbox("Country of Residence", ["United States", "India", "United Kingdom", "Jordan", "Brazil", "Spain", "Egypt", "New Zealand", "Other"])
    relation = st.selectbox("Who is completing this form?", ["Self", "Parent", "Health care professional", "Others"])

    # 🧠 Medical & Family History
    jaundice = st.selectbox("Had jaundice as an infant?", ["Yes", "No"])
    used_app_before = st.selectbox("Used autism screening app before?", ["Yes", "No"])
    family_pdd = st.selectbox("Family history of PDD (for adult)?", ["Yes", "No"])
    autism_diagnosed = st.selectbox("Has autism diagnosis (for child)?", ["Yes", "No"])

    # 🧪 AQ-10 Scores
    st.markdown("### AQ-10 Screening Scores (0 = No, 1 = Yes)")
    a_scores = {}
    for i in range(1, 11):
        a_scores[f"A{i}_Score"] = st.selectbox(f"A{i}_Score", [0, 1], key=f"A{i}")

    # 📊 AQ-10 result score (Total)
    result_score = sum(a_scores.values())  # You could also let user set this
    st.write(f"**AQ-10 Result Score:** {result_score}")

    # 👇 Prepare full feature input
    input_dict = {
        'age': age,
        'gender': gender.lower(),
        'ethnicity': ethnicity,
        'jaundice': 1 if jaundice == "Yes" else 0,
        'used_app_before': 1 if used_app_before == "Yes" else 0,
        'country_of_res': country,
        'relation': relation,
        **a_scores
    }

    # Choose model-specific column
    if age >= 18:
        input_dict['family_pdd'] = 1 if family_pdd == "Yes" else 0
    else:
        input_dict['autism'] = 1 if autism_diagnosed == "Yes" else 0

    input_df = pd.DataFrame([input_dict])

    # 🤖 Predict
    if st.button("Predict"):
        if age >= 18:
            prediction = model_adult.predict(input_df)[0]
            proba = model_adult.predict_proba(input_df)[0][1]
            model_type = "Adult"
        else:
            prediction = model_child.predict(input_df)[0]
            proba = model_child.predict_proba(input_df)[0][1]
            model_type = "Child"

        label = "YES (Likely Autism)" if prediction == 1 else "NO (Unlikely Autism)"
        st.success(f"Prediction using {model_type} model: **{label}**")
        st.info(f"Prediction confidence: **{proba:.2f}**")
