import numpy as np
import pandas as pd
from PythonFunctions.env import host, user ,password
from sklearn.model_selection import train_test_split
import math
import os


def get_connection(db, user = user, host = host, password = password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    
    
    
def get_all_zillow_data():
    '''
    This function gets the zillow data needed to predict single unit properities.
    '''
    file_name = 'zillow.csv'
    if os.path.isfile(file_name):
        return pd.read_csv(file_name)
    
    else:
        query = '''
        select * 
        from predictions_2017
        left join properties_2017 using(parcelid)
        left join airconditioningtype using(airconditioningtypeid)
        left join architecturalstyletype using(architecturalstyletypeid)
        left join buildingclasstype using(buildingclasstypeid)
        left join heatingorsystemtype using(heatingorsystemtypeid)
        left join propertylandusetype using(propertylandusetypeid)
        left join storytype using(storytypeid)
        left join typeconstructiontype using(typeconstructiontypeid)
        where latitude is not null and longitude is not null

                '''
    df = pd.read_sql(query, get_connection('zillow'))  
    
     #replace white space with nulls
    df = df.replace(r'^\s*$', np.NaN, regex=True)
    
    df.to_csv(file_name, index = False)
    return df




def split_for_model(df):
    '''
    This function take in the telco_churn data acquired,
    performs a split into 3 dataframes. one for train, one for validating and one for testing 
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=321)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=231)
    
    print('train{},validate{},test{}'.format(train.shape, validate.shape, test.shape))
    return train, validate, test





#remove nulls and columns based on %
##############################################################
##############################################################
##############################################################
def nulls_by_col(df):
    num_missing = df.isnull().sum()
    print(type(num_missing))
    rows = df.shape[0]
    prcnt_miss = num_missing / rows * 100
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 'percent_rows_missing': prcnt_miss})
    return cols_missing

def nulls_by_row(df):
    num_missing = df.isnull().sum(axis=1)
    prcnt_miss = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': prcnt_miss})\
    .reset_index()\
    .groupby(['num_cols_missing', 'percent_cols_missing']).count()\
    .rename(index=str, columns={'index': 'num_rows'}).reset_index()
    return rows_missing



def get_nulls(df):
    col =  nulls_by_col(df)
    row =  nulls_by_row(df)
    
    return col, row


def drop_null_columns(df , null_min , col_missing = 'percent_rows_missing'):
    cols = get_nulls(df)[0]
    for i in range(len(cols)):
        if cols[col_missing][i] >= null_min:
            df = df.drop(columns = cols.index[i])
        
    return df

def drop_null_rows(df , percentage):
    min_count = int(((100-percentage)/100)*df.shape[1] + 1)
    df = df.dropna(axis=0, thresh = min_count)
        
    return df

def drop_nulls(df, axis, percentage):
    if axis == 0:
        df = drop_null_rows(df, percentage)   
    else:
        df = drop_null_columns(df, percentage)
    return df
##############################################################
##############################################################
##############################################################
def zillow_engineering(zillow_df):
    zillow_df['taxrate'] = round(zillow_df['taxamount']/zillow_df['taxvaluedollarcnt'] * 100 ,2)
    zillow_df['transactiondate'] = pd.to_datetime(zillow_df['transactiondate'],dayfirst=True)
    zillow_df['transactionmonth'] = zillow_df['transactiondate'].dt.month
    zillow_df['log10price'] = np.log10(zillow_df['taxvaluedollarcnt'])

    
    return zillow_df


