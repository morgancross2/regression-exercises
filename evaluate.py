# standard ds imports
import numpy as np
import pandas as pd

# for visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# for modeling and evaluation
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression

def plot_residuals(y_train, yhat):
    '''
    This function takes in the actual and predicted values and 
    plots them. 
    '''
    df = pd.DataFrame({'y_train':y_train,'yhat':yhat})

    
    df['residuals'] = df.y_train - df.yhat
    sns.scatterplot(data=df, x=df.y_train, y=df.residuals)
    plt.xlabel('x = y_train')
    plt.ylabel('y = residuals')
    plt.title('OLS linear model residuals')
    plt.show()
    

def regression_errors(y_train, yhat):
    '''
    This function takes in the actual (y_train) and predicted (yhat) values
    and returns the MSE, SSE, RMSE, TSS, and EES values
    '''
    df = pd.DataFrame({'y_train':y_train,'yhat':yhat})
    df['baseline'] = df.y_train.mean()
    
    MSE = mean_squared_error(df.y_train, df.yhat)
    SSE = MSE*len(df)
    RMSE = mean_squared_error(df.y_train, df.yhat, squared=False)
    TSS = mean_squared_error(df.y_train, df.baseline)*len(df)
    EES = TSS - SSE
        
    return MSE, SSE, RMSE, TSS, EES


def baseline_mean_errors(y_train):
    '''
    This function takes in the actual values (y_train) and returns
    the MSE, SSE, and RMSE for the baseline model.
    '''
    df = pd.DataFrame({'y_train':y_train})
    df['baseline'] = df.y_train.mean()
    
    MSE = mean_squared_error(df.y_train, df.baseline)
    SSE = MSE*len(df)
    RMSE = mean_squared_error(df.y_train, df.baseline, squared=False)
    
    return MSE, SSE, RMSE


def better_than_baseline(y_train, yhat):
    '''
    This function takes in the predicted and actual values and returns
    if the SSE, RMSE and R2 values are better for the model or baseline.
    '''

    MSE, SSE, RMSE, TSS, EES = regression_errors(y_train, yhat)

    MSE_baseline, SSE_baseline, RMSE_baseline = baseline_mean_errors(y_train)

    df = pd.DataFrame({'y_train':y_train,'yhat':yhat})
    df['baseline'] = df.y_train.mean()
    r2_baseline = r2_score(df.y_train, df.baseline)
    r2 = r2_score(df.y_train, df.yhat)

    if SSE < SSE_baseline:
        print(f'''The model SSE performs better than the baseline.
            Baseline SSE: {SSE_baseline}
            Model SSE: {SSE}''')
    else:
        print(f'''The baseline SSE performs better than the model.
            Baseline SSE: {SSE_baseline}
            Model SSE: {SSE}''')
    print()
        
    if RMSE < RMSE_baseline:
        print(f'''The model RMSE performs better than the baseline.
            Baseline RMSE: {RMSE_baseline}
            Model RMSE: {RMSE}''')
    else:
        print(f'''The baseline RMSE performs better than the model.
            Baseline RMSE: {RMSE_baseline}
            Model RMSE: {RMSE}''')
    print()
        
    if r2 > r2_baseline:
        print(f'''The model R2 performs better than the baseline.
            Baseline R2: {r2_baseline}
            Model R2: {r2}''')
    else:
        print(f'''The baseline R2 performs better than the model.
            Baseline R2: {r2_baseline}
            Model R2: {r2}''')