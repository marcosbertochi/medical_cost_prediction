import streamlit as st
import pickle
import pandas as pd
import numpy as np
from MedicalCost import MedicalCosts


st.title('Medical Cost Prediction')

st.write("""Please fill out the customer information in order to provide medical costs prediction.
            All fields are mandatory.
	        This prediction aims to support medical insurance proposal""")


def main():

    # following lines create boxes in which user can enter data required to make prediction

    name = st.text_input("Full Name")
    sex = st.selectbox('Sex',("male","female"))
    age = st.number_input("Age",18,100)
    height = st.number_input("Height (cm)",1,200)
    weight = st.number_input("Weight (kg)",1,200)
    bmi = np.round(( weight / (height/100)**2),2)
    children = st.slider('Children',0,5)
    region = st.selectbox('US Region Residence',("southeast","southwest","northeast","northwest"))
    smoker = st.selectbox('Smoking Habits',("no","yes"))
    dict = {'sex':sex, 'age':age,'bmi':bmi,'children':children, 'region':region, 'smoker':smoker}

    df_raw = pd.DataFrame(dict,index=[0])

    # when 'Predict' is clicked, make the prediction and store it
    if st.button("Predict"):

        #load pickled model
        model = pickle.load(open('model/model_medical_cost_prediction.pkl', 'rb'))

        #instantiate class
        pipeline = MedicalCosts()

        #cleaning
        df = pipeline.data_cleaning(df_raw)

        #feature engineering
        df1 = pipeline.feature_engineering(df)

        #preparation
        df2 = pipeline.data_preparation(df1)

        #predict
        result = pipeline.get_prediction(model, df2)

        st.success('The estimative to medical cost is {}'.format(np.round(result,2)))

        proposal = st.checkbox('Do you want to elaborate a proposal?')

        if proposal:
            st.write("a")

if __name__=='__main__':
    main()
