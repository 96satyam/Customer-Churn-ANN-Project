import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Loading the model

model = tf.keras.models.load_model('churn_model.h5')

# Loading the encoders and scaler

with  open('gender_label_encoder.pkl','rb') as file:
    gender_label_encoder = pickle.load(file)

with  open('onehotencoder.pkl','rb') as file:
    onehotencoder = pickle.load(file)

with  open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)


## Streamlit app

st.title('Customer Churn Prediction App')

# user inputs

geography = st.selectbox('Geography', onehotencoder.categories_[0])
gender = st.selectbox('Gender', gender_label_encoder.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_credit_card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is Active Member', [0,1])


# Preparing the input data

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender' :[gender_label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
})

# One_hot encoding the geography column

geo_encoded = onehotencoder.transform([[geography]]).toarray()

geo_encoded_df = pd.DataFrame(geo_encoded, columns = onehotencoder.get_feature_names_out(['Geography']))

# Concatenating the input data with the encoded geography column

input_data = pd.concat([input_data.reset_index(drop = True), geo_encoded_df], axis = 1)

# Scaling the input data

scaled_input_data = scaler.transform(input_data)

# Making the prediction

prediction = model.predict(scaled_input_data)
prediction_prob = prediction[0][0]

st.write(f'Prediction Probability: {prediction_prob: .2f}')

if prediction_prob > 0.5 :
    st.write("Customer is likely to churn.")
else:
    st.write("Customer is not likely to churn.")