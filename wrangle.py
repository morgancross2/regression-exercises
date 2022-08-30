import numpy as np
import pandas as pd
import os
import env
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


def wrangle_zillow():
    '''
    This function checks if the zillow data is saved locally. 
    If it is not local, this function reads the zillow data from 
    the CodeUp MySQL database and return it in a DataFrame.
    '''
    
    # Acquire
    filename = 'zillow.csv'
    if os.path.isfile(filename):
        df = pd.read_csv(filename).iloc[:,1:]
    else:
        q = '''SELECT bedroomcnt, 
                    bathroomcnt, 
                    calculatedfinishedsquarefeet,
                    taxvaluedollarcnt,
                    yearbuilt,
                    taxamount,
                    fips
                 FROM properties_2017
                 WHERE propertylandusetypeid = 261
             ;'''
        df = pd.read_sql(q, conn('zillow'))
        
        df.to_csv(filename)
        
        
    # Prepare
    cols = ['bedroomcnt', 'bathroomcnt', 'calculatedfinishedsquarefeet','taxvaluedollarcnt','taxamount']
    
    for col in cols:
        q1,q3 = df[col].quantile([.25,.75])
        iqr = q3-q1
        upper = q3 + 1.5*iqr
        lower = q1 - 1.5*iqr

        df = df[(df[col] > lower) & (df[col] < upper)]
    
    df.columns = ['beds', 'baths', 'sqft', 'tax_value', 'built', 'taxes', 'location']
    df = df.drop(index=df[(df.sqft < 300)].index)
    
    return df

def split_data(df):
    '''
    Takes in a dataframe and target (as a string). Returns train, validate, and test subset 
    dataframes with the .2/.8 and .25/.75 splits to create a final .2/.2/.6 split between datasets
    '''
    train, test = train_test_split(df, test_size = .2, random_state=123)
    train, validate = train_test_split(train, test_size = .25, random_state=123)
    
    return train, validate, test


def impute_mode(train, validate, test, col):
    '''
    Takes in train, validate, and test as dfs, and column name (as string) and uses train 
    to identify the best value to replace nulls in embark_town
    
    Imputes the most_frequent value into all three sets and returns all three sets
    '''
    imputer = SimpleImputer(strategy='most_frequent')
    imputer = imputer.fit(train[[col]])
    train[[col]] = imputer.transform(train[[col]])
    validate[[col]] = imputer.transform(validate[[col]])
    test[[col]] = imputer.transform(test[[col]])
    
    return train, validate, test


def scale_data(X_train, y_train, X_val, y_val, X_test, y_test):
    '''
    This function takes in train, val, test datasets and returns the 
    MinMaxScalar values in new dataframes.
    '''
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=['beds','baths','sqft','built','taxes','location'])
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=['beds','baths','sqft','built','taxes','location'])
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=['beds','baths','sqft','built','taxes','location'])

#     train_scaled['built'] = train.built
#     train_scaled['location'] = train.location
#     val_scaled['built'] = val.built
#     val_scaled['location'] = val.location
#     test_scaled['built'] = test.built
#     test_scaled['location'] = test.location
    
    return X_train_scaled, X_val_scaled, X_test_scaled