import pandas as pd
import numpy as np
import streamlit as st
import pickle

# Load the pre-trained model
model = pickle.load(open('calories_model.sav', 'rb'))
#set the page
st.set_page_config(page_title='Calories Burned Predictor',layout='centered')
st.title('Calories Burned Prediction App')
st.write('Enter your details below to estimate the calories burnt during your workout.')

#create the input fields
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender",["Male","Female"])
    age = st.slider("Age", 1, 100, 25)
    height = st.slider("Height (cm)", 100, 250, 170)
    weight = st.slider("Weight (kg)", 30, 200, 70)

with col2:
    duration = st.slider("Duration of Exercise(minutes)",1,60,30)
    heart_rate = st.slider("Heart rate",60,200,100)
    body_temp = st.slider("Body Temperature (Celsius)", 35.0, 42.0, 37.0)


#convert input for model
gender_num = 0 if gender == "Male" else 1

#create the input array
features = np.array([[gender_num, age, height, weight, duration, heart_rate, body_temp]])

#predict calories burned
if st.button("Predict Calories Burned", type="primary", use_container_width=True):
    prediction  = model.predict(features)
    st.success(f"Estimated Calories Burned: {prediction[0]:.2f} kcal")

    # Simple Health Tip based on prediction
    if prediction[0] > 300:
        st.info("Great job! That was a high-intensity workout.")
    else:
        st.info("Keep moving! Every step counts.")



# Setup the sidebar
with st.sidebar:
    st.title("Project Overview")
    st.markdown("""
    ### 📊 About the Model
    This app uses an **Random Forest Regressor** trained on 15,000 workout samples to predict calorie expenditure with **98% accuracy**.
    
    ### 🧑‍💻 Developer
    **[Jahnvi kumai]**
    - [LinkedIn](https://www.linkedin.com/in/jahnvi-kumai-2452a32b5)
    - [GitHub](https://github.com/Jahnvi317)
    
    ### ⚠️ Disclaimer
    Estimates are based on general patterns and may vary based on individual metabolism.
    """)
    
    # Adding a fun interactive element
    st.divider()
    st.info("Tip: Increasing your heart rate by 10 bpm can significantly impact your burn rate!")
