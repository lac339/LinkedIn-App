import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open('lr.model.pkl', 'rb') as file:
    model = pickle.load(file)

def sent_app():
    st.title('LinkedIn Usage Prediction')

    # User inputs
    income = st.slider('Income Level (1-9)', 1, 9, 5)
    education = st.slider('Education Level (1-8)', 1, 8, 4)
    parent = st.selectbox('Are you a parent?', ['No', 'Yes'])
    married = st.selectbox('Are you married?', ['No', 'Yes'])
    gender = st.selectbox('Gender', ['Female', 'Male'])
    age = st.number_input('Age', min_value=1, max_value=100, value=30)

    # Convert categorical inputs to numerical
    parent = 1 if parent == 'Yes' else 0
    married = 1 if married == 'Yes' else 0
    gender = 0 if gender == 'Female' else 1

    # Make a prediction
    input_data = np.array([[income, education, parent, married, gender, age]])
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0, 1]

    # Display the results
    st.write(f'Prediction: {"LinkedIn User" if prediction == 1 else "Non-LinkedIn User"}')
    st.write(f'Probability: {probability:.4f}')

# Run the app
if __name__ == '__main__':
    sent_app()
