#suicide prediction program
import time
'''
suicide prediction program is working 
            but 
will take time so dont quit in middle
'''

# import all parts as module from src
from src import DataProcessing
from src.CorrelationMatrix import CorrMatrix
from src.DataSplitting import DataSplit
from src.FeatureImportance import featuring_importance
import src.TuningWithGridSearchCV as gscv
import src.TuningWithRandomizedSearchCV as rscv
# from src.AccuracyBarGraph import AccuracyPlot


# import modules
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import json
start = time.time()

# data loading, checking, cleaning and encoding
data = DataProcessing.process()

'''
- Data Cleaning nd Encoding
- Corrlation Matrix
- Splitting the data into training and testing
- Feature importance
'''

CorrMatrix(data)

X, y, X_train, X_test, y_train, y_test = DataSplit(data)

featuring_importance(X, y)

#Dictionary to store accuracy results of different algorithms
accuracyDict = {}

'''
- Tuning
'''

# Tuning with GridSearchCV
gscv.GridSearch(X_train, X_test, y_train, y_test, accuracyDict)

# Tuning with RandomizedSearchCV
rscv.RandomizedSearch(X_train, X_test, y_train, y_test, accuracyDict)

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