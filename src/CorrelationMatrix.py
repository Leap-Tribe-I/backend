# importing the required modules
import seaborn as sns
import matplotlib.pyplot as plt

def CorrMatrix(data):
    # correlation matrix
    corr = data.corr()
    # print("\n")
    # print("Correlation Matrix:\n")
    # print(corr)
    # print("\n")
    # f, ax = plt.subplots(figsize=(9, 9))
    # sns.heatmap(corr, vmax=.8, square=True, annot=True)
    # plt.show()
    # plt.savefig('matrix.png')
    return corr