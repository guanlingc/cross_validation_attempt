import pandas as pd
import sqlite3

class DataLoader:
    '''This Class serves to load data from an address named data_path
    Parameter:
    data_path: location of the data file (str) 

    Methods/Functions:
    check_data_path: returns the data path of this class
    initiate_local_connection: starts a connection to the data file
    get_data: extracts data based on a SQL syntax
    '''
    def __init__(self, data_path):
        self.data_path = data_path

    def check_data_path(self):
        '''This function allows users to check on the data file they are extracting data from
        '''
        print('DataPath is ' + self.data_path)
        return self.data_path
    
    def initiate_local_connection(self):
        '''This function defines "conn" as the connection object to the data file'''
        conn = sqlite3.connect(self.data_path)
        print('Local Connection Successful')
        return conn
    
    def get_data(self, sql_query, conn):   
        ''' This function extracts information from the data file via a SQL syntax and
        into a pandas dataframe.
        Parameter:
        sql_query: SQL syntax for what data you want to extract from the data file
        conn: which data file that you want to extract from
        '''
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()  
        df = pd.DataFrame(results)
        print('Data Successfully Obtained')
        print('The shape of the data is ' + str(df.shape))
        conn.close()
        return df
    

    