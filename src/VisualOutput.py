import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def final_plot(log, kn, dis, rand, boosting, bagging):
    X_axis = ['log_reg', 'knn', 'dictree', 'rand_for', 'boost', 'bag']
    bar_width = np.arange(6)
    actual_vals = [log[0], kn[0], dis[0], rand[0], boosting[0], bagging[0]]
    predicted_vals = [log[1], kn[1], dis[1], rand[1], boosting[1], bagging[1]]
    plt.bar(bar_width, actual_vals, width =0.45, align='edge', label="Actual Values")
    plt.bar(bar_width + 0.45, predicted_vals,width=0.45, align='edge', label="Predicted Values")
    # plt.bar(kn,height=21, label="KNN")
    # plt.bar(kn,height=21, label="KNN")
    # plt.bar(dis,height=21, label="Dicision")
    # plt.bar(dis,height=21, label="Dicision")
    # plt.bar(rand,height=21, label="Rand_For")
    # plt.bar(rand,height=21, label="Rand_For")
    # plt.bar(boosting,height=21, label="Boost")
    # plt.bar(boosting,height=21, label="Boost")
    # plt.bar(bagging,height=21, label="Bagging")
    # plt.bar(bagging,height=21, label="Bagging")
    # plt.bar(stacker,height=21, label="Stacking")
    plt.xticks(bar_width, X_axis)
    plt.legend()
    plt.xlabel("Prediction Model")
    plt.ylabel("No. of Predictions")
    plt.show()