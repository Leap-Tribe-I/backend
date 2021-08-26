import os
import time
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from src import DataProcessing
from src.CorrelationMatrix import CorrMatrix
from src.FeatureImportance import featuring_importance

def suicide():
    start = time.time()

    # setting Path to the current working directory
    path = os.getcwd() + "/models/"

    # processing the data input by user
    X_test = DataProcessing.process()


    # creating a dictionary where binary blob of every model will be stored to their corresponding names
    model_dictionary = {}
    # model_name = ['knn', 'log', 'dt', 'rf', 'boost', 'bag', 'stack', 'rand_log', 'rand_knn', 'rand_dt','rand_rf', 'rand_ada','rand_bag','rand_stack']

    # Creating a list of all the files in models directory
    dir_list = os.listdir(path)

    # iterating through the dir_list which contains filenames of all the saved models
    for dir in dir_list:
        if dir == "dnn_pkl" or dir == "test.py":
            continue
        # storing the binary blob and the model name to the model_dictionary
        with open(path+dir, 'rb') as f:
            model_name = dir.partition("p")[0]
            model_dictionary[model_name] = pickle.load(f)

    # defining the data directory, it will contain datafiles input by users
    data_dir = os.getcwd() + "/data/"

    # getting X_test as the input by user (y_test is temporary only for testing of accuracy scores)
    # X_test = pd.read_csv(data_dir+"xtest", index_col=0)
    # y_test = pd.read_csv(data_dir+"ytest", index_col=0)

    # creating a list to store all the prediciton values made by our models
    y_pred_class = []
    
    # iterating through model_dictionary which contains all our saved models
    for key in model_dictionary:
        # appending the prediciton values to y_pred_class list
        y_pred_class.append(model_dictionary[key].predict(X_test))
        print(key)

    # converting y_pred_class to a dataFrame to join it for Correlation Matrix
    y_pred_class_df = pd.DataFrame(y_pred_class).transpose()
    y_pred_class_df.columns = [key for key in model_dictionary]
    print(y_pred_class_df)

    # creating a list to store accuracy socres of our predictions (only for testing)
    # acc_score = []
    # iterating through y_pred_class and checking it with y_test to acertain the accuracy score
    #  and append it to the acc_score list
    # for j in y_pred_class[0]: 
    #     acc_score.append(accuracy_score(y_test, y_pred_class[j]))
    # print(acc_score)

    # creating a correlation matrix between features 
    corrdat = pd.concat([X_test, y_pred_class_df], axis=1, join='inner')
    CorrMatrix(corrdat)

    # FeatureImportance Graph
    featuring_importance(X_test, y_pred_class_df)
    end = time.time()
    print("Time Taken: ", end-start)