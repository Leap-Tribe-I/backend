# import modules needed
import os
import pandas as pd
from sklearn import preprocessing

def process():
    data = load_n_check()
    data = clean(data)
    data = encode(data)
    return data

def load_n_check():
    # data loading
    # enter the location of your input file
    # input_location = input("Enter your input file location (CSV/Excel/Json Allowed): ")
    input_location = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/upload_dir/MHDi.xlsx"
    # check if the file exists
    while not os.path.isfile(input_location):
        print("File does not exist")
        exit()
    # Check input and read file
    if(input_location.endswith(".csv")):
        data = pd.read_csv(input_location)
    elif(input_location.endswith(".xlsx")):
        data = pd.read_excel(input_location)
    elif(input_location.endswith(".json")):
        data = pd.read_json(input_location)
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
    print("Data Loaded and Checked")
    return data

def clean(data):
    # data Cleaning
    # total = data.isnull().sum()
    # precentage = (total/len(data))*100
    # missing_data = pd.concat([total, precentage], axis=1, keys=['Total', 'Precentage'])
    # print("Missing Data:\n")
    # print(missing_data)
    # print("\n")
    # drop unnecessary columns
    if '_id' in data:
        data = data.drop(['_id'], axis=1)
    elif 'Timestamp' in data:
        data = data.drop(['Timestamp'], axis=1)
    # print("\n")
    # print("Dataset afterdropping columns:\n")
    # print(data.head())
    print("Data Cleaned")
    return data

def encode(data):
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
    print("Data Encoded")
    return data