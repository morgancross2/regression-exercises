import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
from sklearn.linear_model import LinearRegression


def explore_cats(train, cats, target):
    '''
    This function takes:
            train = train DataFrame
            cats = category columns (as a list of strings)
            target = target variable (as a string)
    prints value counts for each category in each column
    '''
    for col in cats:
        print(col)
        print(train[col].value_counts())
        print(train[col].value_counts(normalize=True)*100)
        sns.countplot(x=col, data=train)
        plt.title(col+' counts')
        plt.show()
    
        sns.barplot(data=train, x=col, y=target)
        rate = train[target].mean()
        plt.axhline(rate, label= 'average ' + target + ' rate')
        plt.legend()
        plt.title(target+' rate by '+col)
        plt.show()
    
        alpha = 0.05
        o = pd.crosstab(train[col], train[target])
        chi2, p, dof, e = stats.chi2_contingency(o)
        result = p < alpha
        print('P is less than alpha: '+result.astype('str'))
        print('------------------------------------------------------------')

def explore_nums(train, nums):
    '''
    This function takes in:
            train = train DataFrame
            nums = numerical columns (as a list of strings)
    '''
    for col in nums:
        sns.histplot(x=col, data=train)
        plt.show()
        
def plot_chi(train, cats):
    alpha = 0.05
    for col in cats:
        o = pd.crosstab(train[col], train[target])
        chi2, p, dof, e = stats.chi2_contingency(o)
        plt.plot(train[col], chi2)
    plt.show()    
    
    
def select_kbest(X_train, y_train, k_features):
    '''
    This function takes in X_train, y_train, and the number of features
    to select and returns the names of the selected features using SelectKBest
    from sklearn. 
    '''
    kbest = SelectKBest(f_regression, k=k_features)
    kbest.fit(X_train, y_train)
    
    print(X_train.columns[kbest.get_support()].tolist())
    
    
def select_rfe(X_train, y_train, k_features):
    '''
    This function takes in X_train, y_train, and the number of features
    to select and returns the names of the selected features using Recursive
    Feature Elimination from sklearn. 
    '''
    model = LinearRegression()
    rfe = RFE(model, n_features_to_select=k_features)
    rfe.fit(X_train, y_train)
    
    print(X_train.columns[rfe.support_].tolist())
    
    
def select_sfs(X_train, y_train, k_features):
    '''
    This function takes in X_train, y_train, and the number of features
    to select and returns the names of the selected features using Sequential
    Feature Selector from sklearn. 
    '''
    model = LinearRegression()
    sfs = SequentialFeatureSelector(model, n_features_to_select=k_features)
    sfs.fit(X_train, y_train)
    
    print(X_train.columns[sfs.support_].tolist())