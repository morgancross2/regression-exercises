import numpy as np
import pandas as pd
import os
import env
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


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
        q1 = '''SELECT bedroomcnt, 
                    bathroomcnt, 
                    calculatedfinishedsquarefeet,
                    taxvaluedollarcnt,
                    yearbuilt,
                    taxamount,
                    fips
                 FROM properties_2016
                 WHERE propertylandusetypeid = 261
             ;'''
        q2 = '''SELECT bedroomcnt, 
                    bathroomcnt, 
                    calculatedfinishedsquarefeet,
                    taxvaluedollarcnt,
                    yearbuilt,
                    taxamount,
                    fips
                 FROM properties_2017
                 WHERE propertylandusetypeid = 261
             ;'''
        df1 = pd.read_sql(q1, conn('zillow'))
        df2 = pd.read_sql(q2, conn('zillow'))
        df = pd.concat([df1,df2])

        df.to_csv(filename)
        
        
    # Prepare
    df = df.dropna(how='any')
    df.bedroomcnt = df.bedroomcnt.astype('int64')
    df.calculatedfinishedsquarefeet = df.calculatedfinishedsquarefeet.astype('int64')
    df.taxvaluedollarcnt = df.taxvaluedollarcnt.astype('int64')
    df.yearbuilt = df.yearbuilt.astype('int64')
    df.fips = df.fips.astype('int64')
    
    df.columns = ['beds', 'baths', 'sqft', 'price', 'built', 'taxes', 'location']
    df = df.drop(index=df[(df.sqft < 300)].index)
    
    return df

def split_data(df, target):
    '''
    Takes in a dataframe and target (as a string). Returns train, validate, and test subset 
    dataframes with the .2/.8 and .25/.75 splits to create a final .2/.2/.6 split between datasets
    '''
    train, test = train_test_split(df, test_size = .2, random_state=123, stratify=df[target])
    train, validate = train_test_split(train, test_size = .25, random_state=123, stratify=train[target])
    
    return train, validate, test