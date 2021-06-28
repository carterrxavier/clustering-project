import numpy as np
import pandas as pd
from PythonFunctions.env import host, user ,password
from sklearn.model_selection import train_test_split
import geopy.distance
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
                                        random_state=765)
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
    
    zillow_df['latitude'] = zillow_df['latitude'].astype(str)
    zillow_df['longitude'] = zillow_df['longitude'].astype(str)
    
    
    for i in range(len(zillow_df)):
        zillow_df['latitude'][i].replace('.','')
        zillow_df['longitude'][i].replace('.','')
        
        split1 = zillow_df['latitude'][i][:2]
        split2 = zillow_df['latitude'][i][2:-2]
        new = split1 + '.' + split2
        zillow_df['latitude'][i] = new
        
        split1 = zillow_df['longitude'][i][:4]
        split2 = zillow_df['longitude'][i][4:-2]
        new = split1 + '.' + split2
        zillow_df['longitude'][i] = new
        
    zillow_df['latitude'] = round(zillow_df['latitude'].astype(float), 6)
    zillow_df['longitude'] = round(zillow_df['longitude'].astype(float), 6)
                
    return zillow_df

cities = []
#---------LA County Major Cities--------
LosAngeles    = [34.052234, -118.243684] #0
Palmdale      = [34.579434, -118.243684] #1
Lancaster     = [34.686785, -118.154163] #2
Santa_Clarita = [34.391664, -118.542586] #3
Long_Beach    = [33.770050, -118.193739] #4
Glendale      = [34.142507, -118.255075] #5
Pasadena      = [34.147784, -118.144515] #6
Ponoma        = [34.055103, -117.749990] #7
Torrance      = [33.835849, -118.340628] #8
Malibu        = [34.025921, -118.779757] #9
#-------Orange County Major Cities-----
Anahiem       = [33.835293, -117.914503] #10
Santa_Ana     = [33.745573, -117.867833] #11
Irvine        = [33.683947, -117.794694] #12
Newport_Beach = [33.618910, -117.928946] #13
Huntington    = [33.660297, -117.999226] #14
#------Ventura County Major Cities -----
Simi_Valley   = [34.055103, -117.749990] #15
Thousand_oaks = [34.170560, -118.837593] #16
Oxnard        = [34.197504, -119.177051] #17

cities.extend([
    LosAngeles,  
    Palmdale,
    Lancaster,    
    Santa_Clarita, 
    Long_Beach,
    Glendale,     
    Pasadena,      
    Ponoma,   
    Torrance,      
    Malibu,       
    Anahiem,       
    Santa_Ana,     
    Irvine,        
    Newport_Beach, 
    Huntington,    
    Simi_Valley,   
    Thousand_oaks, 
    Oxnard])

#uses geophy to get distance between 2 points in miles
def distance(lat1, lon1, lat2, lon2):
    cor1 = (lat1, lon1)
    cor2 = (lat2, lon2)
    return geopy.distance.distance(cor1, cor2).miles;


# checks the current house location to see which city is the closest       
def get_closest(houselat,houselon):
    results = []
    city = 0 
    for i in cities:
        result = distance(i[0], i[1], houselat, houselon)
        results.append([city ,result])
        city = city + 1 
    results.sort(key= lambda x:x[1])
    return results[0]

 
def get_city_and_distance(df):
    #initialize columns first
    df['closestcity'] = 0
    df['distancefromcity'] = 0
    
    for i in range(len(df)):
        city_and_distance = get_closest(df.latitude[i], df.longitude[i])
        df['closestcity'][i] = city_and_distance[0]
        df['distancefromcity'][i] = city_and_distance[1]
    return df

    



    
    
    
    
