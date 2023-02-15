import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

import warnings
warnings.filterwarnings("ignore")

np.set_printoptions(suppress=True)

seed = 42

#------------------------------------------------------------

def mannwhitney_report(group1, group2):
    '''
    This function takes in two groups (columns), and will perform a mannwhitneyu test on them and print out 
    the test statistic and p-value, as well as determine if the p-value is lower than a predetermined (.05) alpha
    '''
    t, p = stats.mannwhitneyu(group1, group2)

    alpha = .05
    seed = 42

    print(f'T-Statistic = {t:.4f}') 
    print(f'p-value     = {p}')

    print('Is p-value < alpha?', p < alpha)
    
#------------------------------------------------------------

def ind_ttest_report(group1, group2):
    '''
    This function takes in two groups (columns), and will perform an independent t-test on them and print out 
    the t-statistic and p-value, as well as determine if the p-value is lower than a predetermined (.05) alpha
    '''
    t, p = stats.ttest_ind(group1, group2, equal_var=False)

    alpha = .05
    seed = 42

    print(f'T-statistic = {t:.4f}') 
    print(f'p-value     = {p}')

    print('Is p-value < alpha?', p < alpha)
    
#------------------------------------------------------------

def pearsonr_report(group1, group2):
    '''
    This function takes in two groups (columns), and will perform a pearsonr test on them and print out 
    the test statistic and p-value, as well as determine if the p-value is lower than a predetermined (.05) alpha
    '''
    corr, p = stats.pearsonr(group1, group2)

    alpha = .05
    seed = 42

    print(f'Correlation = {corr:.4f}') 
    print(f'p-value     = {p}')

    print('Is p-value < alpha?', p < alpha)
    
#------------------------------------------------------------

def spearmanr_report(group1, group2):
    '''
    This function takes in two groups (columns), and will perform a spearman r test on them and print out 
    the test statistic and p-value, as well as determine if the p-value is lower than a predetermined (.05) alpha
    '''
    corr, p = stats.spearmanr(group1, group2)

    alpha = .05
    seed = 42

    print(f'Correlation = {corr:.4f}') 
    print(f'p-value     = {p}')

    print('Is p-value < alpha?', p < alpha)
    
#------------------------------------------------------------

def chi2_report(df, col, target):
    '''
    This function is to be used to generate a crosstab for my observed data, and use that the run a chi2 test, and generate the report values from the test.
    '''
    alpha = .05
    seed = 42
    
    observed = pd.crosstab(df[col], df[target])
    
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    
    print('Observed Values\n')
    print(observed.values)
    
    print('---\nExpected Values\n')
    print(expected.astype(int))
    print('---\n')

    print(f'chi^2 = {chi2:.4f}') 
    print(f'p     = {p}')

    print('Is p-value < alpha?', p < alpha)
    
#------------------------------------------------------------

def chi_simple(group1, group2):
    
    alpha = .05
    seed = 42
    
    observed = pd.crosstab(group1, group2)
    
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    
    print(f'chi^2 = {chi2:.4f}')
    print(f'p.    = {p}')
    
    print('Is p-value < alpha?', p < alpha)

#------------------------------------------------------------

def anova_report(group1, group2, group3, group4, group5):
    '''
    This function takes in multiple groups (columns), and will perform an ANOVA test on them and print out 
    the f statistic and p-value, as well as determine if the p-value is lower than a predetermined (.05) alpha
    '''
    f, p = stats.f_oneway(group1, group2, group3, group4, group5)
    alpha = .05
    seed = 42

    print(f'f-statistic = {f:.4f}') 
    print(f'p-value     = {p}')

    print('Is p-value < alpha?', p < alpha)
    
#------------------------------------------------------------

def viz_1(train):
    
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})

    sns.lineplot(data=train, x='heart_disease', y='bmi')

    plt.title('Heart Disease and Body Mass Index')
    plt.xlabel('Heart Disease Risk')
    plt.ylabel('Body Mass Index')
    
    plt.show()

#------------------------------------------------------------

def viz_2(train):
    
    fig, ax = plt.subplots(figsize=(7,6))
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})

    bplot = sns.countplot(x='physically_active', hue='heart_disease', data=train)
    ax.bar_label(bplot.containers[0], padding= 6)

    plt.title('Physical Activity and Heart Disease Risk')
    plt.xlabel('Physically Active?')
    plt.ylabel('Heart Disease Risk')
    
    plt.show()

#------------------------------------------------------------

def viz_3(train):
    
    sns.set_style("whitegrid")

    sns.lineplot(data=train, x='physical_health', y='heart_disease')

    plt.title('Physical Health Levels Compared to Heart Disease Risk')
    plt.xlabel('Physical Health')
    plt.ylabel('Heart Disease Risk')
    
    plt.show()

#------------------------------------------------------------

def viz_4(train):
    
    sns.set_style("whitegrid")

    sns.lineplot(x='physical_health', y='mental_health', data=train)
    plt.title('Physical Health and Mental Health')
    plt.xlabel('Physical Health (In the past 30 days)')
    plt.ylabel('Mental Health (In the past 30 days)')
    
    plt.show()
    
#------------------------------------------------------------

def viz_5(train):

    sns.set_style("whitegrid")

    sns.lineplot(x='diabetic', y='bmi', data=train)
    
    plt.title('Do Diabetes and Bmi have a Linear Relationship?')
    plt.xlabel('Is the Patient Diabetic?')
    plt.ylabel('Body Mass Index')
    
    plt.show()

#------------------------------------------------------------



#------------------------------------------------------------



#------------------------------------------------------------



#------------------------------------------------------------
