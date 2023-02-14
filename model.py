import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr, spearmanr
from scipy import stats
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

import warnings
warnings.filterwarnings("ignore")

np.set_printoptions(suppress=True)

seed = 42
    
#--------------------------------------------------------------

def subset_df(df, stratify=None, seed=42):
    '''
    This function takes in a DataFrame and splits it into train, validate, test subsets for our modeling phase. Does not take in a stratify option.
    '''
    train, val_test = train_test_split(df, train_size=.6, random_state=seed)
    validate, test = train_test_split(val_test, train_size=.5, random_state=seed)
        
    return train, validate, test

#--------------------------------------------------------------

def xy_subsets(train, validate, test, target):
    '''
    This function will separate each of my subsets for the dataset (train, validate, and test) and split them further into my x and y subsets for modeling. When running this, be sure to assign each of the six variables in the proper order.
    '''  
    seed = 42
    
    X_train = train.drop(columns=[target])
    y_train = train[target]

    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]

    X_test = test.drop(columns=[target])
    y_test = test[target]
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test

#--------------------------------------------------------------

def inertial_dampening(df, cols, num=11):
    
    inertia = []
    seed = 42

    for n in range(1, num):
    
        kmeans = KMeans(n_clusters=n, random_state=seed)
    
        kmeans.fit(df[cols])
    
        inertia.append(kmeans.inertia_)
        
    results_df = pd.DataFrame({'n_clusters': list(range(1,num)),
              'inertia': inertia})

    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    sns.relplot(data=results_df, x='n_clusters', y='inertia', kind='line', marker='x')

    plt.xticks(range(1, num))
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Change in inertia as number of clusters increase')
    plt.show()

#--------------------------------------------------------------

def clusters_bmi_diabetic(df):
    '''
    This function takes in a dataframe, and creates and fits a clustering model on it, while adding that cluster to the dataframe and renaming the column and dropping the columns created to make the cluster.
    '''
    kmeans = KMeans(n_clusters=3, random_state=seed)
    
    kmeans.fit(df[['bmi', 'diabetic_Yes', 'physically_active_Yes']])
    df['scaled_clusters'] = kmeans.predict(df[['bmi', 'diabetic_Yes', 'physically_active_Yes']])

    sns.relplot(data=df, x='fixed_acidity', y='volatile_acidity', hue='scaled_clusters')
    plt.show()
    
    df = df.rename(columns= {'scaled_clusters': 'acidity_areas'})
    df = df.drop(columns=['fixed_acidity', 'volatile_acidity'])
    
    return df, kmeans

#--------------------------------------------------------------

def cluster_val_test(df, model, cols, name):
    '''
    This function takes in a dataframe and a clustering model to predict off of the already fit model, and creates a column in the given dataframe, as well as renaming the column and dropping the two that the cluster were created from.
    '''
    df1 = df.copy()
    
    df1[name] = model.predict(df1[cols])
    df1 = df1.drop(columns=cols)
    
    return df1

#--------------------------------------------------------------

def make_baseline(df, name, col):
    '''
    This function is used to create a column within the dataframe to make a baseline column, and then calculate the baseline accuracy.
    '''
    seed = 42
    
    df[name] = df[col].value_counts().idxmax()    

    base = (df[col] == df[name]).mean()
    
    print(f'Baseline Accuracy is: {base:.3}')
    
#--------------------------------------------------------------

def rf_gen(X_train, y_train, X_validate, y_validate):
    '''
    This function will create a dataframe of Random Forest models of varying max_depths and 
    compare the differences from the train and validate sets and return the dataframe.
    '''
    metrics = []
    
    seed = 42

    for i in range(1, 16):
        
        rf = RandomForestClassifier(max_depth=i, min_samples_leaf=3, n_estimators=200, random_state=42)
        rf = rf.fit(X_train, y_train)
        
        in_sample_accuracy = rf.score(X_train, y_train)
        out_of_sample_accuracy = rf.score(X_validate, y_validate)
        output = {
            "max_depth": i,
            "train_accuracy": in_sample_accuracy,
            "validate_accuracy": out_of_sample_accuracy
            }
    
        metrics.append(output)
    
    df = pd.DataFrame(metrics)
    df["difference"] = df.train_accuracy - df.validate_accuracy
    
    return df

#--------------------------------------------------------------

def rf_model(X_train_scaled, y_train, X_validate_scaled, y_validate):
    '''
    This funtion will take in four dataframes, the X_train, y_train, X_validate, and y_validate, and create and fit a random forest model to predict the train and validate accuracy on said model.
    '''
    rf = RandomForestClassifier(max_depth=5, min_samples_leaf=5, n_estimators=200, random_state=42)
    rf = rf.fit(X_train_scaled, y_train)
    
    print(f'Train accuracy is: {rf.score(X_train_scaled, y_train):.2f}')
    
    print('-----\n')

    print(f'Validate accuracy is: {rf.score(X_validate_scaled, y_validate):.2f}')
    
    return 
    
#--------------------------------------------------------------

def dectree_gen(X_train, y_train, X_validate, y_validate):
    '''
    This function will create a dataframe of Decision Tree models of varying max_depths and 
    compare the differences from the train and validate sets and return the dataframe. 
    '''
    metrics = []
    
    seed = 42

    for i in range(1, 20):
        
        dectree = DecisionTreeClassifier(max_depth=i, random_state=42)
        dectree = dectree.fit(X_train, y_train)

        in_sample_accuracy = dectree.score(X_train, y_train)
        out_of_sample_accuracy = dectree.score(X_validate, y_validate)
        output = {
            "max_depth": i,
            "train_accuracy": in_sample_accuracy,
            "validate_accuracy": out_of_sample_accuracy
            }
    
        metrics.append(output)
    
    df = pd.DataFrame(metrics)
    df["difference"] = df.train_accuracy - df.validate_accuracy
    
    return df

#--------------------------------------------------------------

def dectree_model(X_train_scaled, y_train, X_validate_scaled, y_validate):
    '''
    This funtion will take in four dataframes, the X_train, y_train, X_validate, and y_validate, and create and fit a decision tree model to predict the train and validate accuracy on said model.
    '''
    dectree = DecisionTreeClassifier(max_depth=4, random_state=42)
    dectree = dectree.fit(X_train_scaled, y_train)
    
    print(f'Train accuracy is: {dectree.score(X_train_scaled, y_train):.2f}')
    
    print('-----\n')
    
    print(f'Validate accuracy is: {dectree.score(X_validate_scaled, y_validate):.2f}')

#--------------------------------------------------------------

def knn_gen(X_train, y_train, X_validate, y_validate):
    '''
    This function will create a dataframe of KNN models of varying n_neighbors and compare the differences from 
    the train and validate sets and return the dataframe.
    '''
    metrics = []

    seed = 42
    
    for i in range(1, 16):
        
        knn = KNeighborsClassifier(n_neighbors=i, weights='uniform')
        knn = knn.fit(X_train, y_train)

        in_sample_accuracy = knn.score(X_train, y_train)
        out_of_sample_accuracy = knn.score(X_validate, y_validate)
        output = {
            "max_depth": i,
            "train_accuracy": in_sample_accuracy,
            "validate_accuracy": out_of_sample_accuracy
            }
    
        metrics.append(output)
    
    df = pd.DataFrame(metrics)
    df["difference"] = df.train_accuracy - df.validate_accuracy
    
    return df

#--------------------------------------------------------------

def xg_boost1(X_train, y_train, X_validate, y_validate, X_test, y_test, max_depth, scale_pos_weight, learning_rate):
    results = []
    for md in max_depth:
        for sw in scale_pos_weight:
            for lr in learning_rate:
                clf_xgb = xgb.XGBClassifier(objective='binary:logistic',
                                            random_state=42,
                                            max_depth=md,
                                            scale_pos_weight=sw,
                                            learning_rate=lr,
                                            subsample=.9,
                                            colsample_bytree=.5,
                                            n_jobs=10)
                clf_xgb.fit(X_train, y_train, verbose=False)
                # Accuracy and recall for the training data
                y_train_pred = clf_xgb.predict(X_train)
                train_accuracy = accuracy_score(y_train, y_train_pred)
                train_recall = recall_score(y_train, y_train_pred)
                # Accuracy and recall for the validate data
                y_validate_pred = clf_xgb.predict(X_validate)
                validate_accuracy = accuracy_score(y_validate, y_validate_pred)
                validate_recall = recall_score(y_validate, y_validate_pred)
                # Accuracy and recall for the test data
                y_test_pred = clf_xgb.predict(X_test)
                test_accuracy = accuracy_score(y_test, y_test_pred)
                test_recall = recall_score(y_test, y_test_pred)
                # Append results to the list
                results.append({'scale_pos_weight': sw,
                                'learning_rate': lr,
                                'max_depth': md,
                                'Train Accuracy': train_accuracy,
                                'Train Recall': train_recall,
                                'Validate Accuracy': validate_accuracy,
                                'Validate Recall': validate_recall,
                                'Test Accuracy': test_accuracy,
                                'Test Recall': test_recall})
                # Print results
                print(f"max_depth: {md}, scale_pos_weight: {sw}, learning_rate: {lr}, Train Recall: {train_recall:.4f}, Validate Recall: {validate_recall:.4f}, Test Recall: {test_recall:.4f}")
    # Convert results list to DataFrame and return
    results_df = pd.DataFrame(results)
    return results_df

#--------------------------------------------------------------

def plot_feature_imp(X_train, y_train):
    
    seed=42
    
    rf = RandomForestClassifier()
    rf_model = rf.fit(X_train, y_train)
    rf_model.feature_importances_
    
    importances = pd.DataFrame(X_train.columns, columns=['features'])
    importances['feature_importances'] = rf_model.feature_importances_
    imp = importances.sort_values(by='feature_importances', ascending=False)
    
    sns.barplot(y=imp['features'], x=imp['feature_importances'])
    
    plt.show()

#--------------------------------------------------------------
