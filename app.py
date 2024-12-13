import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE

#Read in Data
s = pd.read_csv('social_media_usage.csv')

# Create Function to Clean
def clean_sm(x):
    return np.where(x == 1, 1, 0)

# Create and Clean DataFrame
ss = s[['income', 'educ2', 'age']].apply(pd.to_numeric, errors='coerce')
ss['sm_li'] = clean_sm(s['web1h'])
ss['female'] = clean_sm(s['gender'] == 2)
ss['married'] = clean_sm(s['marital'] == 1)
ss['parent'] = clean_sm(s['par'] == 1)

# Handle Missing Values
ss[['income', 'educ2', 'age']] = ss[['income', 'educ2', 'age']].where(
    ss[['income', 'educ2', 'age']] <= [9, 8, 98], np.nan
)

# Drop rows with NaN values
ss.dropna(inplace=True)

#Set Values
y = ss["sm_li"]
x = ss[["educ2", "income", "age", "married", "female", "parent"]]

x_train, x_test, y_train, y_test = train_test_split(x.values,
                                                    y, stratify = y,
                                                    test_size=.2,
                                                    random_state=444)


#Train Model
lr = LogisticRegression(class_weight='balanced')
lr.fit(x_train, y_train)
model = lr.fit(x_train, y_train)

#Rebalance
smote = SMOTE(random_state=4)
x_resampled, y_resampled = smote.fit_resample(x_train, y_train)
lr.fit(x_resampled, y_resampled)
model1 = lr.fit(x_resampled, y_resampled)

#Evaluate model using test data
y_pred = lr.predict(x_test)

#create App
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
    prediction = model1.predict(input_data)
    probability = model1.predict_proba(input_data)[0, 1]

    # Display the results
    st.write(f'Prediction: {"LinkedIn User" if prediction == 1 else "Non-LinkedIn User"}')
    st.write(f'Probability: {probability:.4f}')

# Run the app
if __name__ == '__main__':
    sent_app()
