import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import classification_report,confusion_matrix,recall_score, precision_score
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


import warnings
warnings.filterwarnings('ignore')




def get_risiduals(df ,act, pred):
    df['risiduals'] = act - pred
    df['baseline_risiduals'] = act - act.mean()
    return df

def plot_residuals(act, pred, res, baseline):
    plt.figure(figsize=(16,9))
    plt.subplot(221)
    plt.title('Residuals')
    res.hist()
    plt.subplot(222)
    plt.title('Baseline Residuals')
    baseline.hist()
    
    
    ax = plt.subplot(223)
    ax.scatter(act, pred)
    ax.set(xlabel='actual', ylabel='prediction')
    ax.plot(act, act,  ls=":", color='black')
    
    ax = plt.subplot(224)
    ax.scatter(act, res)
    ax.set(xlabel='actual', ylabel='residual')
    ax.hlines(0, *ax.get_xlim(), ls=":",color='black')
    
    plt.show()
    
def regression_errors(y, yhat):
    sse = ((y-yhat) ** 2).sum()
    mse = sse / y.shape[0]
    rmse = math.sqrt(mse)
    ess = ((yhat - y.mean())**2).sum()
    tss = ((y - y.mean())**2).sum()
    r_2 = (ess/tss)
    
    return sse, mse, rmse, ess, tss, r_2

def baseline_errors(y):
    sse_baseline = ((y - y.mean()) ** 2).sum()
    mse_baseline = sse_baseline / y.shape[0]
    rmse_baseline = math.sqrt(mse_baseline)
    return sse_baseline, mse_baseline, rmse_baseline


def better_than_baseline(y, yhat):
    print('Baseline:{}'.format(baseline))
    run = regression_errors(y,yhat)[2] < baseline
    return run

def select_kbest(X,y,top):
    f_selector = SelectKBest(f_regression, top)
    f_selector.fit(X,y.logerror)
    result = f_selector.get_support()
    f_feature = X.loc[:,result].columns.tolist()
    return f_feature

def select_rfe(X, y, n):
    lm = LinearRegression()
    rfe = RFE(lm, n)
    X_rfe = rfe.fit_transform(X,y)
    mask = rfe.support_
    rfe_feautures = X.loc[:,mask].columns.tolist()
    return rfe_feautures

def get_logbin(X_train,number):
    X_train = X_train[X_train['logbin_pred'] > number]
    return X_train


def get_baseline(df, target):
    Baseline = DummyClassifier(strategy = 'constant' , constant= df[target].mode())
    return Baseline


    
    
def get_t_test(t_var, df, target, alpha):
    '''
        This method will produce a 2 tailed t test  equate the p value to the alpha to determine whether the null hypothesis can be rejected.
    '''
    for i in t_var:
        t, p = stats.ttest_ind(df[i],df[target], equal_var=False)
        print('Null Hypothesis: {} has no correlation to {}'.format(i,target))
        print('Alternative hypothesis:  {} has correlation to {} '.format(i, target))
        if p < alpha:
            print('p value {} is less than alpha {} , we reject our null hypothesis'.format(p,alpha))
        else:
            print('p value {} is not less than alpha {} , we fail to reject our null hypothesis'.format(p,alpha))
        print('-------------------------------------')
        
def get_pearsons(con_var, target, alpha, df):
     for i in con_var:
        t, p = stats.pearsonr(df[i],df[target])
        print('Null Hypothesis: there is not linear correlation between {} and {} '.format(i, target))
        print('Alternative hypothesis:  {} has linear correlation  to {} '.format(i,target))
        if p < alpha:
            print('p value {} is less than alpha {} , we reject our null hypothesis'.format(p,alpha))
        else:
            print('p value {} is not less than alpha {} , we  fail to reject our null hypothesis'.format(p,alpha))
        print('-------------------------------------')
        
def get_chi_test(chi_list, df, alpha, target):
    '''
    This method will produce a chi-test contengency, and equate the p value to the alpha to determine whether the null hypothesis can be rejected.
    ''' 
    for i in chi_list:
        observed = pd.crosstab(df[i], df[target])
        print(observed)
        chi2, p, degf, expected = stats.chi2_contingency(observed)
        print('Null Hyothesis: {} and {} are independent to eachother'.format(i, target))
        print('Alternative hypothesis: {} and {} have dependency to one another'.format(i,target))
        if p < alpha:
            print('p value {} is less than alpha {} , we reject our null hypothesis'.format(p,alpha))
        else:
            print('p value {} is not less than alpha {} , we  fail to reject our null hypothesis'.format(p,alpha))
        print('-------------------------------------')
        

def get_model_results(X_train, y_train, X, y, target, model='linear', normalize=False, alpha = 0, power = 0, degree=2 , graph = False):
    results = y.copy()
    
    if model == "linear":
        lm = LinearRegression(normalize=normalize)
        lm.fit(X_train, y_train)  
        results['pred'] = lm.predict(X)
        
    elif model == 'lasso':
        lasso = LassoLars(alpha=alpha)
        lasso.fit(X_train, y_train)
        results['pred'] = lasso.predict(X)
        
    elif model == 'glm':
        glm = TweedieRegressor(power=power, alpha=alpha)
        glm.fit(X_train, y_train)
        results['pred'] = glm.predict(X)
    elif model == 'poly':
        pf = PolynomialFeatures(degree=degree)
        
        X_train_degree2 = pf.fit_transform(X_train)
        X_degree_2 = pf.transform(X)
        
        
        lm = LinearRegression(normalize=True)
        lm.fit(X_train_degree2, y_train)  
        results['pred'] = lm.predict(X_degree_2)
        
    results = get_risiduals(results, y[target], results.pred)
    rmse =    regression_errors(y[target],results.pred)[2]
    r_2  =    regression_errors(y[target],results.pred)[5]

    print('r2 Score:  {}'.format(r_2))
    print('RMSE Score: {}'.format(rmse)) 
  
   

    if graph == True:
        plot_residuals(results[target], results.pred, results.risiduals, results.baseline_risiduals)
    
    return results, rmse



def train_validate_results(model, X_train,y_train, X_validate, y_validate, drivers=None):

    '''
    this function prints the accuracy of the model passed in
   
    '''
    if drivers==None:
        model.fit(X_train, y_train)
        t_pred = model.predict(X_train)
        v_pred = model.predict(X_validate)
        print('Train model Accuracy: {:.5f} % | Validate model accuracy: {:.5f} % '.format(model.score(X_train, y_train) * 100, model.score(X_validate, y_validate) * 100))
    else:
        model.fit(X_train[drivers], y_train)
        t_pred = model.predict(X_train[drivers])
        v_pred = model.predict(X_validate[drivers])
        print('Train model Accuracy: {:.5f} % | Validate model accuracy: {:.5f} % '.format(model.score(X_train[drivers], y_train) * 100, model.score(X_validate[drivers], y_validate) * 100))
    return t_pred, v_pred
    

    
        
def test_results(model, X_test, y_test, X_train, y_train, drivers=None):
    '''
    this function prints the accuracy, recall and precision of the model passed in
    '''
    if drivers == None:
        model.fit(X_train, y_train)
        t_pred = model.predict(X_test)
        print('Test model Accuracy: {:.5f} %'.format(model.score(X_test, y_test) * 100))
    else:
        model.fit(X_train[drivers], y_train)
        t_pred = model.predict(X_test[drivers])
        print('Test model Accuracy: {:.5f} %'.format(model.score(X_test[drivers], y_test) * 100))
    return t_pred
    
    