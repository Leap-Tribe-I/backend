# importing modules
from sklearn import preprocessing

def dce(data):
    DataCleaning(data)
    DataEncoding(data)
    return data

def DataCleaning(data):
    # data Cleaning
    # total = data.isnull().sum()
    # precentage = (total/len(data))*100
    # missing_data = pd.concat([total, precentage], axis=1, keys=['Total', 'Precentage'])
    # print("Missing Data:\n")
    # print(missing_data)
    # print("\n")
    # drop unnecessary columns
    if 'Timestamp' in data:
        data = data.drop(['Timestamp'], axis=1)
    # print("\n")   
    # print("Dataset afterdropping columns:\n")
    # print(data.head())
    return data

def DataEncoding(data):
    # data encoding
    labelDictionary = {}
    for feature in data:
        le = preprocessing.LabelEncoder()
        le.fit(data[feature])
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        data[feature] = le.transform(data[feature])
        # Get labels
        labelKey = 'label_' + feature
        labelValue = [*le_name_mapping]
        labelDictionary[labelKey] =labelValue

    # print(labelDictionary)
    # for key, value in labelDictionary.items():     
    #     print(key, value)

    # print("\n")
    # print("Dataset after encoding:\n")
    # print(data.head())
    # print("\n")

    # output the encoded data
    # data.to_csv('_encoded.csv')
    # print("\n")
    # print("Encoded data saved as: _encoded.csv")
    return data