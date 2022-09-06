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
    
    The prepare portion of this function removes outliers via 
    quantiles. 
        - Renames the features
        - Feature engineers a readable location
        - Feature engineers year_built into decade bins
        - Feature engineers tax_value percentiles(quadrants), split for location
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
    # remove outliers
    cols = ['bedroomcnt', 'bathroomcnt', 'calculatedfinishedsquarefeet','taxvaluedollarcnt','taxamount']
    
    for col in cols:
        q1,q3 = df[col].quantile([.25,.75])
        iqr = q3-q1
        upper = q3 + 1.5*iqr
        lower = q1 - 1.5*iqr

        df = df[(df[col] > lower) & (df[col] < upper)]
    
    # rename columns
    df.columns = ['beds', 'baths', 'sqft', 'tax_value', 'built', 'taxes', 'city']
    # create readable location column from city/fips codes
    df['location'] = df.city.map({6037: 'Los Angeles', 6059: 'Orange', 6111:'Ventura'})
    
    dummies = pd.get_dummies(df.city)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(columns='city')
    
    df.columns = ['beds', 'baths', 'sqft', 'tax_value', 'built', 'taxes', 'location', 'los_angeles', 'orange', 'ventura']
    df.los_angeles = df.los_angeles.astype(int)
    df.orange = df.orange.astype(int)
    df.ventura = df.ventura.astype(int)

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

def feature_engineer(df):
    
    # Creates decades
    df['decade'] = pd.cut(df.built, 
                         bins=[1800,1850,1900,1910,1920,1930,1940,1950,1960,1970,1980,1990,2000,2010,2020], 
                         labels=['1800', '1850', '1900', '1910', '1920', '1930', '1940', '1950', '1960', '1970', '1980', '1990', '2000', '2010'])
    df.decade = df.decade.astype(int)
    
    # Creates location percentiles based on train split's taxes feature, random_state = 123
    dist = []
    for index, house in df.iterrows():
        if house.location == 'Los Angeles':
            if house.taxes <= 2243.965:
                dist.append('1')
            elif house.taxes <= 3560.36:
                dist.append('2')
            elif house.taxes <= 5263.4:
                dist.append('3')
            else:
                dist.append('4')
        elif house.location == 'Orange':
            if house.taxes <= 2695.16:
                dist.append('1')
            elif house.taxes <= 4203.56:
                dist.append('2')
            elif house.taxes <= 6013.24:
                dist.append('3')
            else:
                dist.append('4')
        elif house.location == 'Ventura':
            if house.taxes <= 2578.57:
                dist.append('1')
            elif house.taxes <= 3969.28:
                dist.append('2')
            elif house.taxes <= 5553.95:
                dist.append('3')
            else:
                dist.append('4')
        else:
            dist.append('ukn')
    df['location_percentile'] = dist
    df.location_percentile = df.location_percentile.astype(int)
    
    return df


def vis_scaler (scaler, df, cols_to_scale, bins=10):
    fig, axs = plt.subplots(len(cols_to_scale),2,figsize=(16,9))
    df_scaled = df.copy()
    df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    
    for (ax1, ax2), col in zip(axs, cols_to_scale):
        ax1.hist(df[col], bins=bins)
        ax1.set(title=f'{col} before scaling', xlabel=col, ylabel='count')
        ax2.hist(df_scaled[col], bins=bins)
        ax2.set(title=f'{col} after scaling', xlabel=col, ylabel='count')
    plt.tight_layout()


def scale_data(train, val, test, cols_to_scale):
    train_scaled = train.copy()
    val_scaled = val.copy()
    test_scaled = test.copy()
    
    scaler = MinMaxScaler()
    scaler.fit(train[cols_to_scale])
    
    train_scaled[cols_to_scale] = pd.DataFrame(scaler.transform(train[cols_to_scale]),
                                               columns = train[cols_to_scale].columns.values).set_index([train.index.values])
    val_scaled[cols_to_scale] = pd.DataFrame(scaler.transform(val[cols_to_scale]),
                                               columns = val[cols_to_scale].columns.values).set_index([val.index.values])
    test_scaled[cols_to_scale] = pd.DataFrame(scaler.transform(test[cols_to_scale]),
                                               columns = test[cols_to_scale].columns.values).set_index([test.index.values])
    
    return train_scaled, val_scaled, test_scaled

