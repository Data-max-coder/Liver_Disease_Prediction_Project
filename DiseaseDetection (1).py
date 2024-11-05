import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the binary classification model (no disease vs disease)
binary_model = joblib.load('Binaryclass_neural_network_model.pkl')

# Load the multiclass classification model for disease type prediction
multiclass_model = joblib.load('multiclass_neural_network_model.pkl')

# Initialize session state for tracking prediction results and inputs
if 'prediction' not in st.session_state:
    st.session_state.prediction = None  # Track the prediction result

if 'features' not in st.session_state:
    st.session_state.features = None  # Track input features

# Initialize session state for each input to retain the values after rerun
if 'age' not in st.session_state:
    st.session_state.age = 0
if 'sex' not in st.session_state:
    st.session_state.sex = 'Male'
if 'albumin' not in st.session_state:
    st.session_state.albumin = 0.0
if 'alanine_aminotransferase' not in st.session_state:
    st.session_state.alanine_aminotransferase = 0.0
if 'bilirubin' not in st.session_state:
    st.session_state.bilirubin = 0.0
if 'cholesterol' not in st.session_state:
    st.session_state.cholesterol = 0.0
if 'gamma_glutamyl_transferase' not in st.session_state:
    st.session_state.gamma_glutamyl_transferase = 0.0
if 'alkaline_phosphatase' not in st.session_state:
    st.session_state.alkaline_phosphatase = 0.0
if 'aspartate_aminotransferase' not in st.session_state:
    st.session_state.aspartate_aminotransferase = 0.0
if 'cholinesterase' not in st.session_state:
    st.session_state.cholinesterase = 0.0
if 'creatinina' not in st.session_state:
    st.session_state.creatinina = 0.0
if 'protein' not in st.session_state:
    st.session_state.protein = 0.0

input_defaults = {
    'patient_name': '',
    'age': 0,
    'sex': 'Male',
    'albumin': 0.0,
    'alanine_aminotransferase': 0.0,
    'bilirubin': 0.0,
    'cholesterol': 0.0,
    'gamma_glutamyl_transferase': 0.0,
    'alkaline_phosphatase': 0.0,
    'aspartate_aminotransferase': 0.0,
    'cholinesterase': 0.0,
    'creatinina': 0.0,
    'protein': 0.0
}

# Sidebar Navigation
st.sidebar.title("Navigation")

# Create custom style for radio buttons and contrast
st.markdown(
    """
    <style>
    .stRadio > div {flex-direction: column;}
    .stRadio label {
        border-radius: 20px; 
        background-color: #e0e0e0; 
        padding: 10px; 
        margin-bottom: 5px;
        width: 100%;
        text-align: center;
    }
    .stRadio label:hover {
        background-color: #d0d0d0;
    }
    .stRadio input:checked + label {
        background-color: #007bff; 
        color: white;
    }
    /* Dark Theme Support */
    .stRadio label {
        transition: background-color 0.3s ease;
    }
    body {
        background-color: #f0f2f6;  /* Light theme background */
    }
    @media (prefers-color-scheme: dark) {
        body {
            background-color: #262730;  /* Dark theme background */
        }
        .stRadio label {
            background-color: #3b3b4f;  /* Darker capsule color for dark theme */
            color: white;
        }
        .stRadio input:checked + label {
            background-color: #42a5f5;  /* Contrast for dark theme */
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

tabs = st.sidebar.radio("Select a page", ["Home", "Liver Disease Detection", "Liver Disease Details"])


# Home Page
if tabs == "Home":
    st.title("Welcome to the Liver Disease Prediction System")
    st.image("liver_jpg.jpg", use_column_width=True)  # Adjust "liver_jpg.jpg" with your filename

    st.write("""
        The liver is one of the most vital organs in the human body, responsible for various functions such as detoxification, protein synthesis, and digestion. 
        However, it can suffer from various diseases like hepatitis, cirrhosis, fibrosis, suspect disease.

        **Common Liver Diseases**:
        - Hepatitis: Inflammation of the liver caused by viruses or toxic substances.
        - Cirrhosis: Scarring of the liver due to long-term damage, often from alcohol abuse or hepatitis.
        - Fibrosis: Occurs when the healthy tissue of your liver becomes scarred and cannot work as well.
        - Suspect: If you suspect liver disease, consult a doctor.

        Regular checkups and monitoring are important for maintaining liver health. Use this app to predict potential liver disease based on your lab results.
    """)

# Liver Disease Detection Page
elif tabs == "Liver Disease Detection":
    st.title("Liver Disease Detection")
    st.write('Input the patient details to get the prediction.')

    # Collect patient name (optional)
    patient_name = st.text_input("Enter Patient Name (Optional)")

    # Create two columns for the input fields
    col1, col2 = st.columns(2)

    # Column 1 Inputs
    with col1:
        age = st.number_input("Age", min_value=0, max_value=100, step=1, value=st.session_state.age)
        st.session_state.age = age
        albumin = st.number_input("Albumin", min_value=0.0, step=0.1, value=st.session_state.albumin)
        st.session_state.albumin = albumin
        alanine_aminotransferase = st.number_input("Alanine Aminotransferase", min_value=0.0, step=0.1, value=st.session_state.alanine_aminotransferase)
        st.session_state.alanine_aminotransferase = alanine_aminotransferase
        bilirubin = st.number_input("Bilirubin", min_value=0.0, step=0.1, value=st.session_state.bilirubin)
        st.session_state.bilirubin = bilirubin
        cholesterol = st.number_input("Cholesterol", min_value=0.0, step=0.1, value=st.session_state.cholesterol)
        st.session_state.cholesterol = cholesterol
        gamma_glutamyl_transferase = st.number_input("Gamma Glutamyl Transferase", min_value=0.0, step=0.1, value=st.session_state.gamma_glutamyl_transferase)
        st.session_state.gamma_glutamyl_transferase = gamma_glutamyl_transferase

    # Column 2 Inputs
    with col2:
        sex = st.selectbox("Sex", ['Male', 'Female'], index=['Male', 'Female'].index(st.session_state.sex))
        st.session_state.sex = sex
        alkaline_phosphatase = st.number_input("Alkaline Phosphatase", min_value=0.0, step=0.1, value=st.session_state.alkaline_phosphatase)
        st.session_state.alkaline_phosphatase = alkaline_phosphatase
        aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase", min_value=0.0, step=0.1, value=st.session_state.aspartate_aminotransferase)
        st.session_state.aspartate_aminotransferase = aspartate_aminotransferase
        cholinesterase = st.number_input("Cholinesterase", min_value=0.0, step=0.1, value=st.session_state.cholinesterase)
        st.session_state.cholinesterase = cholinesterase
        creatinina = st.number_input("Creatinina", min_value=0.0, step=0.1, value=st.session_state.creatinina)
        st.session_state.creatinina = creatinina
        protein = st.number_input("Protein", min_value=0.0, step=0.1, value=st.session_state.protein)
        st.session_state.protein = protein

    # Convert sex to numeric (1: Female, 0: Male)
    sex_numeric = 0 if sex == 'Male' else 1

    # Collect all inputs into a feature list
    features = [age, sex_numeric, albumin, alkaline_phosphatase, alanine_aminotransferase,
                aspartate_aminotransferase, bilirubin, cholinesterase, cholesterol,
                creatinina, gamma_glutamyl_transferase, protein]

    if st.button('Predict'):
        input_data = np.array([features])
        binary_prediction = binary_model.predict(input_data)
        st.session_state.prediction = binary_prediction
        st.session_state.features = features

        if binary_prediction == 0:
            st.write("No Liver Disease Detected!")
        else:
            st.write("Liver Disease Detected!")
            # Predict probabilities using the multiclass model
            disease_probabilities = multiclass_model.predict_proba(input_data)[0]
            disease_types = ['No Disease', 'Suspect Disease', 'Hepatitis', 'Fibrosis', 'Cirrhosis']  # Now includes 'No Disease'

            # Bar plot for disease probabilities
            st.write("Probability of Different Diseases:")
            fig, ax = plt.subplots()
            ax.bar(disease_types, disease_probabilities)
            ax.set_xlabel('Disease Type')
            ax.set_ylabel('Probability')
            st.pyplot(fig)

        # Redirect to new page
        st.rerun()

# Result Page: Display Prediction and Navigation Options
if st.session_state.prediction is not None:
    if st.session_state.prediction == 0:
        st.title("No Liver Disease Detected!")
        st.write("""
            Congratulations! Your liver seems healthy.
            Remember to maintain a balanced diet and avoid excessive alcohol to keep your liver in top shape.
        """)
        st.balloons()  # Celebration effect

        # Options to go back to the input page
        st.write("What would you like to do next?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button('Go Back to Input Page (Cross-check)'):
                # Keep previous inputs and go back to the input page for correction
                st.session_state.prediction = None  # Clear prediction
                st.rerun()
        with col2:
            if st.button('New Prediction'):
                # Clear previous inputs for a fresh prediction
                # st.session_state.prediction = None
                # st.session_state.features = None
                # st.rerun()
                for key in input_defaults:
                    st.session_state[key] = input_defaults[key]
                st.session_state.prediction = None
                st.session_state.features = None
                st.rerun()

    else:
        st.title("Liver Disease Detected")
        st.write("""
            Based on your test results, there are indications of liver disease.
            It is important to consult a doctor for further investigation.

            **Common Symptoms of Liver Disease**:
            - Fatigue
            - Yellowing of the skin and eyes (jaundice)
            - Abdominal pain and swelling
            - Nausea or vomiting
            - Dark urine
            - Chronic fatigue
        """)

        # Options to go back to the input page
        st.write("What would you like to do next?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button('Go Back to Input Page (Cross-check)'):
                # Keep previous inputs and go back to the input page for correction
                st.session_state.prediction = None  # Clear prediction
                st.rerun()
        with col2:
            if st.button('New Prediction'):
                # Clear previous inputs for a fresh prediction
                # st.session_state.prediction = None
                # st.session_state.features = None
                # st.rerun()
                for key in input_defaults:
                    st.session_state[key] = input_defaults[key]
                st.session_state.prediction = None
                st.session_state.features = None
                st.rerun()

# Liver Disease Details Page
elif tabs == "Liver Disease Details":
    st.title("Liver Disease Details")
    st.write("""
        The liver is susceptible to a variety of diseases due to its role in metabolizing substances in the body.
        Below are some of the most common liver diseases:

        **Hepatitis**:
        Inflammation of the liver often caused by viral infections or autoimmune conditions.
        **Symptoms**:
        Abdominal pain, Dark urine, Pale or clay-colored stools, Fatigue, Low-grade fever.

        **Cirrhosis**:
        Chronic damage to the liver results in scar tissue formation and loss of liver function.
        **Symptoms**:
        Fatigue, Easily Bleeding, Loss of appetite, Weight loss, Itchy skin, Swelling in the legs called "EDEMA".

        **Fibrosis**:
        Liver fibrosis occurs when excessive amounts of scar tissue build up in the liver repeatedly.
        **Symptoms**:
        Jaundice, Fatigue, Loss of appetite, Nausea & vomiting, Fever.
    """)
