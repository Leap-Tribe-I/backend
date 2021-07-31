# importing the required modules
import matplotlib.pyplot as plt
import pandas as pd

def AccuracyPlot(accuracyDict):
    # save accuracyDict accuracy Bar Graph to file
    s = pd.Series(accuracyDict)
    s = s.sort_values(ascending=False)
    plt.figure(figsize=(12,8))
    ax = s.plot(kind='bar') 
    for p in ax.patches:
        ax.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.005, p.get_height() * 1.005))
    plt.xlabel('Method')
    plt.ylabel('Percentage')
    plt.title('Success of methods')
    plt.show()
    plt.savefig('Accuracy Graph.png')