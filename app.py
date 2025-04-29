import tempfile
import numpy as np
import scipy
import tensorflow as tf
import streamlit as st
import pandas as pd
import joblib  # or pickle depending on your model
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import seaborn as sns
import nibabel as nib
from streamlit_option_menu import option_menu
from tensorflow.keras.models import load_model as keras_load_model
# from tensorflow.keras.layers import (Input, Conv3D, BatchNormalization, Activation,
#                                      Add, MaxPooling3D, GlobalAveragePooling3D,
#                                      Dropout, Dense)
# import scipy.ndimage
# import tensorflow as tf
# from tensorflow.keras.models import Model


st.set_page_config(layout="wide")
selection = option_menu(
    menu_title=None,  # No title
    options=["About", "EDA", "Prediction"],
    icons=["info-circle", "bar-chart-line", "activity"],  # Some cool icons
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#f0f2f6"},
        "icon": {"color": "black", "font-size": "20px"},
        "nav-link": {
            "font-size": "18px",
            "text-align": "center",
            "margin": "0px",
            "--hover-color": "#eee",
        },
        "nav-link-selected": {"background-color": "#E28979", "color": "white"},
    }
)


# Page: About
if selection == "About":
    st.title("🧠 Autism Detection using Machine Learning Approach")
    st.write("""
   Autism Spectrum Disorder (ASD) is a complex neurodevelopmental condition marked by wide-ranging symptoms in behavior, communication, and cognition. Early and accurate diagnosis plays a critical role in ensuring individuals receive the support they need—but traditional behavioral assessments often fall short due to variability in symptom presentation.

This platform showcases a hybrid approach to ASD detection, combining the power of neuroimaging and behavioral analysis with machine learning. Leveraging both MRI-based brain structure analysis and structured behavioral data from children and adults, this project explores the diagnostic potential of artificial intelligence.

🧪 **What’s Inside:**

A deep learning model trained on 3D MRI scans from the ABIDE dataset using advanced 3D CNNs and cloud-native infrastructure.

Machine learning pipelines on tabular datasets with features like age, gender, ethnicity, and symptom scores, optimized with SMOTE, L1 selection, and classifiers like XGBoost and Random Forest.

Interactive visualizations of autism patterns by age, gender, country, and ethnicity for both children and adults.

By blending clinical insight with cutting-edge AI, we aim to contribute to more objective, scalable, and early detection strategies for autism. Dive in to explore the data, models, and discoveries.
    

## Our Approach

We developed two complementary pipelines for ASD detection:

**MRI-Based Detection**:  
A deep learning model was trained on 3D MRI scans from the ABIDE dataset. Advanced 3D Convolutional Neural Networks (CNNs) were employed, with the MRI volumes preprocessed through voxel intensity normalization and resizing to a consistent shape (80×128×128). The model training was accelerated using GPU infrastructure on Google Cloud Platform, tackling the computational demands of volumetric brain imaging.

**Behavioral Screening-Based Detection**:  
We also built machine learning pipelines based on structured screening data collected from children and adults. This tabular data included features like age, gender, ethnicity, and symptom scores. Models such as Random Forest, Logistic Regression, and XGBoost were trained with preprocessing steps like missing value imputation, SMOTE-based oversampling to handle class imbalance, and L1-based feature selection to enhance model generalizability.

Both pipelines were embedded into an interactive Streamlit application, allowing users to either upload MRI scans for deep learning prediction or input behavioral responses for a machine learning-based assessment. The platform also features visual dashboards analyzing autism trends by demographic factors like gender, country, and ethnicity.

---

## Our Work in Action

Throughout the project, we:

- Built and trained a custom 3D Residual CNN for MRI classification.
- Engineered robust behavioral ML pipelines with hyperparameter tuning and SMOTE balancing.
- Trained MRI models on cloud-native GPU resources using Google Cloud Platform.
- Designed and deployed an integrated Streamlit web application for real-time predictions.
- Created interactive Tableau dashboards to visualize demographic autism trends.

This multi-pronged strategy demonstrates how blending clinical insight with AI tools can lead to more scalable and accurate approaches to ASD detection.
             """)

# Page: EDA
elif selection == "EDA":
    st.title("📊 Exploratory Data Analysis")
    
    # Load your dataset
    df = pd.read_csv("cleaned_adult_data.csv")  # Replace with your actual file

    st.write("### Adult screening dataset Preview")
    st.dataframe(df.head())
   
    df2 = pd.read_csv("cleaned_children_data.csv")  # Replace with your actual file

    st.write("### Children screening dataset Preview")
    st.dataframe(df2.head())

    # st.write("### Correlation Heatmap")
    # plt.figure(figsize=(10,6))
    # sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    # st.pyplot(plt)

    # st.write("### Class Distribution")
    # sns.countplot(data=df, x='Class')  # Update 'Class' with your target column
    # st.pyplot(plt)
    # Embed Dashboard 1: Adults
    # st.subheader("Autism Distribution by Gender in Adults")
    # adult_dashboard_url = "https://public.tableau.com/views/Autismdistributionbygender/Dashboard1?:embed=true&:showVizHome=no"
    # components.iframe(adult_dashboard_url, height=850, width=1000)
    components.html(
    """
    <div style="display: flex; justify-content: center;">
        <iframe 
            src="https://public.tableau.com/views/Autismdistributionbygender/Dashboard1?:embed=true&:showVizHome=no"
            width="1000" 
            height="850" 
            style="border: none;"
        ></iframe>
    </div>
    """,
    height=880
    )
    # Embed Dashboard 2: Children
    # st.subheader("Autism Distribution by Gender in Children")
    # children_dashboard_url = "https://public.tableau.com/views/GenderwiseautismcountinChildren/Dashboard1?:embed=true&:showVizHome=no"
    # components.iframe(children_dashboard_url, height=850, width=1000)
    components.html(
    """
    <div style="display: flex; justify-content: center;">
        <iframe 
            src="https://public.tableau.com/views/GenderwiseautismcountinChildren/Dashboard1?:embed=true&:showVizHome=no"
            width="1000" 
            height="850" 
            style="border: none;"
        ></iframe>
    </div>
    """,
    height=880
)
    

elif selection == "Prediction":
    st.title("🤖 Predict Autism")
    st.markdown(
    """
    <style>
    label {
        font-size: 22px !important;
    }
    h1, h2, h3, h4 {
        font-size: 32px !important;
    }
    div.stButton > button {
        font-size: 20px !important;
        font-weight: bold !important;
        color: white !important;
        background-color: #4CAF50 !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
    }
    div.stButton > button:hover {
        background-color: #45a049 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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
   
    aq10_questions_adult = {
        "A1_Score": "I often notice patterns in things all the time.",
        "A2_Score": "I usually concentrate more on the whole picture than the small details.",
        "A3_Score": "I find it easy to do more than one thing at once.",
        "A4_Score": "If there is an interruption, I can switch back to what I was doing very quickly.",
        "A5_Score": "I find it easy to ‘read between the lines’ when someone is talking to me.",
        "A6_Score": "I know how to tell if someone listening to me is getting bored.",
        "A7_Score": "When I’m reading a story, I find it difficult to work out the characters intentions.",
        "A8_Score": "I like to collect information about categories of things (e.g.types of car,types of bird etc)",
        "A9_Score": "I find it easy to work out what someone is thinking or feeling just by looking at their face.",
        "A10_Score": "I find it difficult to work out people's intentions."
    }

    aq10_questions_child = {
        "A1_Score": "My child notices small sounds when others do not.",
        "A2_Score": "He/she usually concentrates more on the whole picture, rather than the small details.",
        "A3_Score": "In a social group, she/he can easily keep track of several different people's conversations.",
        "A4_Score": "My child finds it easy to go back and forth between different activities.",
        "A5_Score": "She/he doesn't know how to keep a conversation going with his/her peers.",
        "A6_Score": "My child is good at social chit-chat.",
        "A7_Score": "When my child reads a story, he/she finds it difficult to work out the character's intentions or feelings.",
        "A8_Score": "When she/he was in preschool, my child used to enjoy playing games involving pretending with other children",
        "A9_Score": "My child finds it easy to work out what someone is thinking or feeling just by looking at their face.",
        "A10_Score":"My child finds it hard to make new friends."
    }

    # Scoring rules (same for Adult and Child)
    agree_positive = {1, 7, 8, 10}   # Score if Agree
    disagree_positive = {2, 3, 4, 5, 6, 9}  # Score if Disagree

    # Response options
    response_options = ["Definitely Agree", "Slightly Agree", "Slightly Disagree", "Definitely Disagree"]

    # Select target group
    group = st.radio("Screening for:",["Adult","Child"], horizontal = True)

    # Choose correct set of questions
    questions = aq10_questions_adult if group == "Adult" else aq10_questions_child 


    
    st.markdown(f"### AQ-10 Questionnaire for {group}s")
    st.write("Select the most accurate response for each statement:")

    responses = {}
    for i, (key, question) in enumerate(questions.items(), start=1):
     responses[key] = st.selectbox(f"Q{i}: {question}", response_options, key=f"{group}_Q{key}")

    # 🧮 Score calculation
    score = 0
    a_scores = {}

    for key, response in responses.items():
        question_number = int(key[1])  # Extract the number from 'A1_Score' etc.

        if question_number in agree_positive and response in ["Definitely Agree", "Slightly Agree"]:
            score += 1
            a_scores[key] = 1
        elif question_number in disagree_positive and response in ["Definitely Disagree", "Slightly Disagree"]:
            score += 1
            a_scores[key] = 1
        else:
            a_scores[key] = 0

    st.write(f"**AQ-10 {group} Score:** {score} / 10")
    # Prepare full feature input
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
    # if st.button("Predict"):
    #     if age >= 18:
    #         prediction = model_adult.predict(input_df)[0]
    #         proba = model_adult.predict_proba(input_df)[0][1]
    #         model_type = "Adult"
    #     else:
    #         prediction = model_child.predict(input_df)[0]
    #         proba = model_child.predict_proba(input_df)[0][1]
    #         model_type = "Child"

    #     label = "YES (Likely Autism)" if prediction == 1 else "NO (Unlikely Autism)"
    #     st.success(f"Prediction using {model_type} model: **{label}**")
    #     st.info(f"Prediction confidence: **{proba:.2f}**")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Behavioral Model Prediction")

        if st.button("Predict results from screening input"):
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



    with col2:
        def load_mri_model():
          return tf.keras.models.load_model("final_model_best_3.keras") 
 
        
        mri_model = load_mri_model()

        # ✅ Streamlit UI for MRI prediction
        st.subheader("🧠 MRI-Based Model Prediction")
        uploaded_file = st.file_uploader("Upload a 3D MRI Scan (.nii or .nii.gz)", type=["nii", "nii.gz"])
 

        if uploaded_file:
            st.write("📂 File received:", uploaded_file.name)

            if st.button("Predict from MRI"):
               try:
                    # ✅ Get the correct extension
                    file_extension = ".nii.gz" if uploaded_file.name.endswith(".nii.gz") else ".nii"

                    # ✅ Save to temp file with the right extension
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name

                    # ✅ Now load without confusion
                    img = nib.load(tmp_path)
                    data = img.get_fdata()

                    # Normalize voxel intensities [0,1]
                    data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-6)

                    # Resize MRI volume to (80, 128, 128)
                    target_shape = (80, 128, 128)
                    zoom_factors = (
                        target_shape[0] / data.shape[0],
                        target_shape[1] / data.shape[1],
                        target_shape[2] / data.shape[2],
                    )
                    data_resized = scipy.ndimage.zoom(data, zoom=zoom_factors, order=1)

                    # Prepare input for model
                    data_final = np.expand_dims(data_resized, axis=(0, -1)).astype(np.float32)  # Shape: (1, 80, 128, 128, 1)

                    # Predict
                    prediction = mri_model.predict(data_final)[0][0]
                    label = "YES (Likely Autism)" if prediction > 0.5 else "NO (Unlikely Autism)"

                    # Display result
                    st.success(f"MRI Prediction: **{label}**")
                    st.info(f"Prediction confidence: **{prediction:.2f}**")

               except Exception as e:
                    st.error(f"❌ Error during MRI prediction: {e}")

        else:
            st.info("Please upload a 3D MRI scan to proceed.")