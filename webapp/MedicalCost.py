import pickle
import pandas as pd
import numpy as np

class MedicalCosts(object):



    def __init__(self):
        #initiate scalers
        self.home_path = ''
        self.age_scaler =   pickle.load(open(self.home_path+'parameters\\age_scaler.pkl','rb'))
        self.bmi_scaler =   pickle.load(open(self.home_path+'parameters\\bmi_scaler.pkl','rb'))
        self.children_scaler =   pickle.load(open(self.home_path+'parameters\\children_scaler.pkl','rb'))


    def data_cleaning(self,df):

        def outlier_detection(data, column):
            """ Function to remove outliers based on interquartile  intervals
                Args:
                    data:
                    column:

                Return:
                    list:

            """
            q1 = np.percentile(data[column], 25)
            q3 = np.percentile(data[column], 75)

            iqr = q3 - q1

            lo_lim = q1 - 1.5*iqr
            up_lim = q3 + 1.5*iqr

            outliers = [x for x in data[column] if (x > up_lim) | (x<lo_lim)]

            return sorted(outliers)

        #remove bmi outliers
        outliers_bmi = outlier_detection(df,'bmi')
        outliers_bmi
        df = df[~df['bmi'].isin(outliers_bmi)]

        return df

    def feature_engineering(self, df1):

        categ_weight = lambda x: 'underweight' if x < 18.5 else 'normal weight' if x < 25 else 'overweight' if x < 30 else 'obese I' if x < 35 else 'obese II' if x < 40 else 'obese III'
        df1['bmi class'] = df1['bmi'].apply(categ_weight)

        return df1

    def data_preparation(self,df2):

        def dummy_region(df):
            if df['region'].iloc[0] == 'southeast':
                df['region_southeast'] = 1
                df['region_southwest'] = 0
                df['region_northwest'] = 0
                df['region_northeast'] = 0

            elif df['region'].iloc[0] == 'southwest':
                df['region_southeast'] = 0
                df['region_southwest'] = 1
                df['region_northwest'] = 0
                df['region_northeast'] = 0

            elif df['region'].iloc[0] == 'northwest':
                df['region_southeast'] = 0
                df['region_southwest'] = 0
                df['region_northwest'] = 1
                df['region_northeast'] = 0

            elif df['region'].iloc[0] == 'northeast':
                df['region_southeast'] = 0
                df['region_southwest'] = 0
                df['region_northwest'] = 0
                df['region_northeast'] = 1

            df = df.drop(columns='region')

            return df    

        #scalers
        df2['bmi'] = self.bmi_scaler.transform(df2[['bmi']].values)
        df2['children'] = self.children_scaler.transform(df2[['children']].values)
        df2['age'] = self.age_scaler.transform(df2[['age']].values)


        #Binary Encoding - sex
        sex_mapping = {'male':0,'female':1}
        df2['sex'] = df2['sex'].map(sex_mapping)

        #Binary Encoding - smoker
        smoker_mapping = {'no':0,'yes':1}
        df2['smoker'] = df2['smoker'].map(smoker_mapping)

        #OHE - region
        df2 = dummy_region(df2)
    

        #ordinal encoding - BMI class (evaluate to use target encoding next crisp)
        bmi_order_mapping = {'underweight':0,'normal weight':1,'overweight':2, 'obese I':3, 'obese II':4,'obese III':5}
        df2['bmi class'] = df2['bmi class'].map(bmi_order_mapping)

        return df2

    def get_prediction(self, model, df3):

        prediction = model.predict(df3)
        return  np.expm1(prediction)
