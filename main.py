#suicide prediction program
import time
'''
suicide prediction program is working 
            but 
will take time so dont quit in middle
'''

# import all parts as module from src
from src import DataCleaningEncoding
from src.CorrelationMatrix import CorrMatrix
from src.DataSplitting import DataSplit
from src.FeatureImportance import featuring_importance
import src.TuningWithGridSearchCV as gscv
import src.TuningWithRandomizedSearchCV as rscv
import src.DnnClassifier as dc
from src.AccuracyBarGraph import AccuracyPlot

# ignore all warnings
import warnings
warnings.filterwarnings("ignore")

# import modules
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
import json

# data loading
#enter the location of your input file
input_location = input("Enter your input file location: ")
# check if the file exists
while not os.path.isfile(input_location):
    print("File does not exist")
    exit()
# Check input and read file
if(input_location.endswith(".csv")):
    data = pd.read_csv(input_location)
elif(input_location.endswith(".xlsx")):
    data = pd.read_excel(input_location)
else:
    print("ERROR: File format not supported!")
    exit()

# check data
variable = ['family_size', 'annual_income', 'eating_habits', 
            'addiction_friend', 'addiction', 'medical_history', 
            'depressed', 'anxiety', 'happy_currently', 'suicidal_thoughts']
check = all(item in list(data) for item in variable)
if check is True:
    print("Data is loaded")
else:
    print("Dataset doesnot contain: ", variable)
    exit()
start = time.time()
'''
- Data Cleaning nd Encoding
- Corrlation Matrix
- Splitting the data into training and testing
- Feature importance
'''
data = DataCleaningEncoding.dce(data)

CorrMatrix(data)

X, y, X_train, X_test, y_train, y_test = DataSplit(data)

featuring_importance(X, y)

#Dictionary to store accuracy results of different algorithms
accuracyDict = {}

'''
- Tuning
'''

# Tuning with GridSearchCV
# gscv.GridSearch(X_train, X_test, y_train, y_test, accuracyDict)

# Tuning with RandomizedSearchCV
rscv.RandomizedSearch(X_train, X_test, y_train, y_test, accuracyDict)

#using tensorflow nueral network for prediction
# dc.tensorflow_dnn(data,X_train, y_train, X_test, y_test, accuracyDict)

print("accuracyDict:\n")
print(json.dumps(accuracyDict, indent=1))
end = time.time()
'''
- Accuracy Bar Graph
'''

# AccuracyPlot(accuracyDict)

'''
- Modelling
'''

end = time.time()
print("Time taken: ", end - start,"seconds")