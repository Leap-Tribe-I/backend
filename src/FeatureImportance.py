#importing the libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

#Acertaining the feature importance
def featuring_importance(X,y):
    #using ExtraTreesClassifier to acertain important features
    frst = ExtraTreesClassifier(random_state = 0)
    frst.fit(X,y)
    imp = frst.feature_importances_
    stan_dev = np.std([tree.feature_importances_ for tree in frst.estimators_], axis = 0)

    #creating an nparray with decreasing order of importance of features
    indices = np.argsort(imp)[::-1]

    #appending column names to labels list
    labels = []
    for f in range(X.shape[1]):
        labels.append(X.columns[f])

    #ploting feature importance bar graph
    # plt.figure(figsize=(12,8))
    # plt.title("Feature importances")
    # plt.bar(range(X.shape[1]), imp[indices],
    #     color="g", yerr=stan_dev[indices], align="center")
    # plt.xticks(range(X.shape[1]), labels, rotation='vertical')
    # plt.xlim([-1, X.shape[1]])
    # plt.show()
    # plt.savefig('FeatureImportance.png')