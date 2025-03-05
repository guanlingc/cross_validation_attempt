import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    '''This class will contain all of the data cleaning and feature engineering steps'''
    def __init__(self):
        self.scaler = StandardScaler()

# This section contains the functions for post splitting of data  
    def select_features(self, df: pd.DataFrame):
        '''This functions lets you choose the features that you would like to include in the model
        To check on what kind of preprocessing and feature engineering are performed, 
        refer to "clean_data" function'''
        X = df.drop('Survive', axis=1)
        y = df['Survive']
        return X, y
    
    def scale_data(self, X_train, X_test):
        '''This function performs scaling of the test sets using standardscaler
        Takes in X_train and X_test
        
        Returns the scaled version '''
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
    
# This section of code contains all of the preprocessing and feature engineering processes:
# Everything is complied under "clean_data" function
# All feature engineering steps is segregated via type of engineering 
        
    def clean_data(self, df: pd.DataFrame):
        '''This function is a compliation of all the cleaning steps that will be performed for the dataset
        Parameter:
        df: Pandas Dataframe
        
        Returns: 
        cleaned df 
        '''
        renamed_df = self._rename_columns(df)                                       # Rename the columns first
        renamed_df.drop_duplicates(subset='ID', keep ='first', inplace = True)      # Dropping observations with duplicated ID          
        reduced_df = renamed_df.drop(columns=['ID','Favourite_Color'])              # Dropping ID and favourite color as they are redundant
        same_value_df = self._replace_value(reduced_df)                             # 
        BMI_df = self._create_BMI(same_value_df)
        dropped_df = self._drop_columns(BMI_df)
        log_norm_df = self._log_norm_all(dropped_df)
        no_outlier_df = self._remove_outlier_all(log_norm_df)
        label_encoded_df = self._encode_label_all(no_outlier_df)
        ordinal_encoded_df = self._ordinal_encode_all(label_encoded_df)
        return ordinal_encoded_df         
    
    def _rename_columns(self, df: pd.DataFrame):
        '''This is an internal function that renames all of the columns 
        according to the data dictionary given in the assignment write up'''
        renamed_df = df.rename(columns = {
                    0: 'ID',
                    1: 'Survive',
                    2: 'Gender',
                    3: 'Smoke',
                    4: 'Diabetes',
                    5: 'Age',
                    6: 'Ejection_Fraction',
                    7: 'Sodium',
                    8: 'Creatinine',
                    9: 'Platelets',
                    10: 'Creatine_Phosphokinase',
                    11: 'Blood_Pressure',
                    12: 'Hemoglobin',
                    13: 'Height',
                    14: 'Weight',
                    15: 'Favourite_Color',
                    })
        return renamed_df
    
    def _create_BMI(self, df: pd.DataFrame):
        '''This internal function creates the BMI feature using information from Height and Weight 
        using the formula BMI = Weight / Height_squared
        Parameter:
        df: Pandas DataFrame
        
        Returns: 
        BMI: new feature in the Pandas DataFrame'''
        df['BMI'] = round(df['Weight']/((df['Height']/100)**2),1)
        return df

    def _replace_value(self,df: pd.DataFrame):
        """This internal function takes reads a pandas dataframe and the following
        'Survive': Changes all values to 0 and 1 (int)
        'Smoke': Changes all values to No and Yes
        'Ejection_Fraction': Changes all values to Low, Normal and High 
        'Age': Negative values are assumed to be entry errors 
        Parameter:
        df: pandas dataframe
    
        Returns:
        df: pandas dataframe with all values replaced
        """
        df['Survive'].replace(['No','Yes'],['0', '1'], inplace=True)
        df['Smoke'].replace(['NO','YES'],['No','Yes'], inplace=True)
        df['Ejection_Fraction'].replace(['L','N'],['Low','Normal'], inplace=True)
        df['Age'] = df['Age'].abs()
        return df
    
    def _drop_columns(self, df:pd.DataFrame):
        '''This internal function will drop the observations with missing values
        Parameter:
        df: pandas dataframe

        Returns:
        df: pandas dataframe with missing values dropped
        '''
        df = df.dropna(subset=['Creatinine'])
        return df

    def _log_norm_all(self, df: pd.DataFrame):
        ''' This function bundles all of the features that will undergo log normalization
        using the "_log_norm" function 
        '''
        df = self._log_norm(df, 'Platelets')
        df = self._log_norm(df, 'Creatine_Phosphokinase')
        df = self._log_norm(df, 'Blood_Pressure')
        df = self._log_norm(df, 'Creatinine')
        return df
    
    def _remove_outlier_all(self, df: pd.DataFrame):
        ''' This function bundles all of the features that will undergo outlier removal
        using the "_remove_outlier" function 
        '''
        df = self._remove_outlier(df, 'Sodium', .05,0.95)
        df = self._remove_outlier(df, 'log_Platelets', .05,0.95)
        df = self._remove_outlier(df, 'log_Creatine_Phosphokinase', .05,0.95)
        df = self._remove_outlier(df, 'log_Creatinine', .05,0.95)
        return df
    
    def _ordinal_encode_all(self, df: pd.DataFrame):
        ''' This function bundles all of the features that will undergo ordinal encoding
        using the "_ordinal_encode" function 
        '''
        df = self._ordinal_encode(df, 'Diabetes', ['Normal', 'Pre-diabetes', 'Diabetes'])
        df = self._ordinal_encode(df, 'Ejection_Fraction', ['Low', 'Normal', 'High'])
        return df
    
    def _encode_label_all (self, df: pd.DataFrame):
        ''' This function bundles all of the features that will undergo label encoding
        using the "_encode_label" function 
        '''
        df = self._encode_label(df, 'Gender')
        df = self._encode_label(df, 'Smoke')
        df = self._encode_label(df,'Survive')
        return df
    
# This section contains the functions used to perform data preprocessing and feature engineering
    def _remove_outlier(self, df: pd.DataFrame, target_column:str , lower_limit_percentile, upper_limit_percentile):
        '''This function removes outliers from a given feature
        Parameters:
        df: Pandas Dataframe
        target_column: Feature name
        lower_limit_percentile: lower limit of values to keep
        upper_limit_percentile: upper limit of values to keep
        
        Returns:
        df: Feature with outliers removed'''
        lower_lim = df[target_column].quantile(lower_limit_percentile)
        upper_lim = df[target_column].quantile(upper_limit_percentile)
        df = df[(df[target_column] < upper_lim) & (df[target_column] > lower_lim)]
        return df
    
    def _log_norm(self, df: pd.DataFrame, target_column:str):
        """This function takes reads a pandas dataframe creates a log normalization in a new column with the log suffix (log_target_column)
        Parameter:
        df: pandas dataframe
        column_name (str): column to be log normalized

        Returns:
        a new column that stores all of the observations that undergone log normalization
        """
        df[('log_'+ target_column)] = np.log(df[target_column])
        return df
    
    def _ordinal_encode(self, df: pd.DataFrame, target_column: str, ranks=list):
        '''This function performs ordinal encoding
        Parameter:
        df: pandas dataframe
        target_column (str): Feature name to be encoded
        ranks (list): a list of strings to encode in ascending order of ranking 

        Returns:
        df: Feature column is encoded.
        '''
        or_encoder = OrdinalEncoder(categories = [ranks])
        df[target_column] = or_encoder.fit_transform(df[[target_column]])
        return df

    def _encode_label(self, df: pd.DataFrame, target_column: str):
        '''This internal function performs label encoding for a target feature
        Parameter:
        df: pandas dataframe
        target_column (str): Feature name to be encoded

        Returns:
        df: Feature column is encoded.
        '''
        label_encoder = LabelEncoder()
        df[target_column] = label_encoder.fit_transform(df[target_column])
        return df
    
