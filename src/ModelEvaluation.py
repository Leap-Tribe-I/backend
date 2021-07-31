# importing module
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, mean_squared_error, precision_recall_curve

#Tuning and evaluation of models
def evalModel(model, X_test, y_test, y_pred_class):
    acc_score = metrics.accuracy_score(y_test, y_pred_class)
    # print("Accuracy: ", acc_score)
    # print("NULL Accuracy: ", y_test.value_counts())
    # print("Percentage of ones: ", y_test.mean())
    # print("Percentage of zeros: ", 1 - y_test.mean())
    #creating a confunsion matrix
    conmat = metrics.confusion_matrix(y_test, y_pred_class)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    # sns.heatmap(conmat, annot=True)
    # plt.title("Confusion " + str(model))
    # plt.xlabel("predicted")
    # plt.ylabel("Actual")
    # plt.show()

    return acc_score