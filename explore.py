import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

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
    