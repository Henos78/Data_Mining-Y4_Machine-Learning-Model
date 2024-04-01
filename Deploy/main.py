 # Import required libraries
from Deploy import Heart_disease_Logistic.pkl
import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image

 # Load the trained model from pickle file
pickle_in = open("Heart_disease_Logistic.pkl","rb")
final_model=pickle.load(pickle_in)


# Define sex and other choice options
sex_choices=[(0, 'Female'),(1, 'Male')]
cp_choice=[(0,'None'),(1, 'Typical Angina'),(2, 'Atypical Angina'),(3, 'Non-Angina'),(4, 'Asymptomatic')]
fasting_blood_sugar_choices=[(1,'> 120 mg/dl'),((0,'< 120 mg/dl'))]
resting_ecg_choices=[(0, 'Normal'),(1, 'Having ST-T wave abnormality'),(2, 'hypertrophy')]
exercise_induced_angina_choices=[(0, 'No'),(1, 'Yes')]
st_slope_choices=[(1, 'Upsloping'),(2, 'Flat'),(3, 'Down Sloping')]
number_of_vessels_choices=[(0, 'None'),(1, 'One'),(2, 'Two'),(3, 'Three')]
thallium_scan_results_choices=[(3, 'Normal'),(6, 'Fixed Defect'),(7, 'Reversible Defect')]

@st.cache_data
def load_model():
    pickle_in = open("Heart_disease_Logistic.pkl","rb")
    model=pickle.load(pickle_in)
    return model


def predict(age,sex,cp,resting_bp,serum_cholesterol,fasting_blood_sugar,resting_ecg,max_heart_rate,exercise_induced_angina,st_depression,st_slope,number_of_vessels,thallium_scan_results):
    input_data = np.array([float(age), int(sex), int(cp), float(resting_bp), float(serum_cholesterol), int(fasting_blood_sugar), int(resting_ecg), float(max_heart_rate), int(exercise_induced_angina), float(st_depression), int(st_slope), int(number_of_vessels), int(thallium_scan_results)]).reshape(1, -1)
    model = load_model()
    prediction = model.predict(input_data)
    return prediction


def main():
    st.set_page_config(page_title="Heart Disease Prediction ML Model", 
                       page_icon=":bar_chart:", 
                       layout="wide")
    st.title("Heart Disease Prediction ML Model")
    
    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Home":
        home_page()
    elif choice == "About":
        about_page()

def home_page():
    html_temp = """
    <div style="background-color:green;padding:10px">
    <h2 style="color:white;text-align:center;">HEART DISEASE PREDICTION APP </h2>
    <style>body{background-color: #FFFFFF;}</style>
    </div>

    <div>
        <p style="color:white;text-align:center;"> PLEASE FILL  IN ALL THE REQUIRED INFORMATION </p>
        
    </div>
    """
    image = Image.open("image1.jpg")   # Update the file path with your actual image file path
    st.image(image, caption='', use_column_width=True)
    st.markdown(html_temp,unsafe_allow_html=True)

    try:
        age = st.text_input("Age:")
        sex = st.selectbox("Select Sex: [(0, 'Female'),(1, 'Male')]", [choice for choice, _ in sex_choices])
        cp = st.selectbox("Select Chest Pain: [(0,'None'),(1, 'Typical Angina'),(2, 'Atypical Angina'),(3, 'Non-Angina'),(4, 'Asymptomatic')]", [choice for choice, _ in cp_choice])
        resting_bp = st.text_input("Resting BP:")
        serum_cholesterol = st.text_input("Serum Cholesterol Level:")
        fasting_blood_sugar = st.selectbox("Select Fasting Blood Sugar Level:  [(1,'> 120 mg/dl'),((0,'< 120 mg/dl'))]", [choice for choice, _ in fasting_blood_sugar_choices])
        resting_ecg = st.selectbox("Select Resting Electrocardiographic Result: [(0, 'Normal'),(1, 'Having ST-T wave abnormality'),(2, 'hypertrophy')]", [choice for choice, _ in resting_ecg_choices])
        max_heart_rate = st.text_input("Maximum Heart Rate:")
        exercise_induced_angina = st.selectbox("Select Excercise Induced Angina:  [(0, 'No'),(1, 'Yes')]", [choice for choice, _ in exercise_induced_angina_choices])
        st_depression = st.text_input("St Depression:")
        st_slope = st.selectbox("Select Slope Of The Peak Exercise ST Segment: [(1, 'Upsloping'),(2, 'Flat'),(3, 'Down Sloping')]", [choice for choice, _ in st_slope_choices])
        number_of_vessels = st.selectbox("Select Number Of Major Vessels (0-3) Colored By Flourosopy:[(0, 'None'),(1, 'One'),(2, 'Two'),(3, 'Three')] ", [choice for choice, _ in number_of_vessels_choices])
        thallium_scan_results = st.selectbox("Select Thalium Scan Results: [(3, 'Normal'),(6, 'Fixed Defect'),(7, 'Reversible Defect')]", [choice for choice, _ in thallium_scan_results_choices])
        
    except Exception as e:
        st.error("Error occurred: Please enter a valid input {}".format(e))
    
    result = 0
    if st.button("Predict"):
        result=predict(age,sex,cp,resting_bp,serum_cholesterol,fasting_blood_sugar,resting_ecg,max_heart_rate,exercise_induced_angina,st_depression,st_slope,number_of_vessels,thallium_scan_results)
    st.success(' => Here is your result {} (hg/ha).' .format(str(result)))
    st.success(" >> If your result is '0' it means you dont have a heart  disease but if your result is '1' YOU HAVE A HEART DISEASE SO PLEASE CONSULT A DOCTOR NOW! " )

# define a function for the about page  
def about_page():
    html_temp = """
    <div style="background-color:GREY;padding:10px">
    
    
    </div>
    """
    
    """
-> Cardiovascular diseases are the leading cause of death globally, resulted in 17.9 million deaths (32.1%) in 2015, up from 12.3 million (25.8%) in 1990. It is estimated that 90% of CVD is preventable. There are many risk factors for heart diseases that we will take a closer look at.

The main objective of this study is to build a model that can predict the heart disease occurrence, based on a combination of features (risk factors) describing the disease. Different machine learning classification techniques will be implemented and compared upon standard performance metric such as accuracy.

The dataset used for training this model was downloaded from Kaggle.
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    image = Image.open("1.jpg")   # Update the file path with your actual image file path
    st.image(image, caption='', use_column_width=True)

    """
    

Description:

This data set currently contains 303 instances, some of which aren't complete (some features may be missing for a certain instance). In the case that this happens, the instance has been removed. There are 14 relevant features which have been extracted, from a maximum of 76 in the total dataset.

Features information:

1.age - age in years

2.sex - sex(1 = male; 0 = female)

3.chest_pain - chest pain type (1 = typical angina; 2 = atypical angina; 3 = non-anginal pain; 4 = asymptomatic)

4.blood_pressure - resting blood pressure (in mm Hg on admission to the hospital)

5.serum_cholestoral - serum cholestoral in mg/dl

6.fasting_blood_sugar - fasting blood sugar > 120 mg/dl (1 = true; 0 = false)

7.electrocardiographic - resting electrocardiographic results (0 = normal; 1 = having ST-T; 2 = hypertrophy)

8.max_heart_rate - maximum heart rate achieved

9.induced_angina - exercise induced angina (1 = yes; 0 = no)

10.ST_depression - ST depression induced by exercise relative to rest

11.slope - the slope of the peak exercise ST segment (1 = upsloping; 2 = flat; 3 = downsloping)

12.no_of_vessels - number of major vessels (0-3) colored by flourosopy

13.thal - 3 = normal; 6 = fixed defect; 7 = reversable defect

14.num:diagnosis - the predicted attribute - diagnosis of heart disease (angiographic disease status) (Value 0 = < 50% diameter narrowing; Value 1 = > 50% diameter narrowing)

Types of features:

A. Categorical features (Has two or more categories and each value in that feature can be categorised by them): sex, chest_pain

B. Ordinal features (Variable having relative ordering or sorting between the values): fasting_blood_sugar, electrocardiographic, induced_angina, slope, no_of_vessels, thal, diagnosis

C. Continuous features (Variable taking values between any two points or between the minimum or maximum values in the feature column): age, blood_pressure, serum_cholestoral, max_heart_rate, ST_depression

    """
    
    

if __name__=='__main__':
    main()
