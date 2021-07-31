# Suicide-Prediction
Leap 4.0 Major Project

MentalGeek Presents Suicide Prediction Program

## Modularized the Program 
# How to run
- for running download both the src folder and suicide.py file in one directory
- next just run it with python command

# suicide.py file: 
This is the main file and everything will be called here for implementation, only this file will run as the program

# dataCleaningEncoding.py file:
It will clean and run the code

# correlationMatrix.py file:
It will create a corelation matrix and polt a heat map showing the correlation

# dataSplit.py file:
It will split the data set into training and testing

# dataFeaturing.py file:
It will determine the important feature

# tuningGrid.py file:
It will implement different prediction algorithms with GridSearch tuning method

#tuningRand.py file:
It will implement different prediciton algorithms with RandomSearch CV tuning method

# modulEvaluator.py file: 
It will calculate the accuracy of the models

# accuracyPlot.py file:
It will plot the accuracy of all the implemented algorithms in a bar graph

# Output

Correlation Matrix:

| rows and columns   |family_size | annual_income | eating_habits | addiction_friend | ... | depressed |  anxiety | happy_currently | suicidal_thoughts|
|--------------------|------------|---------------|---------------|------------------|-----|-----------|----------|-----------------|------------------|
|family_size          | 1.000000     | -0.042214      | 0.066318         | 0.023710  |...   |0.053648  |0.162366        |-0.050309           |0.080420|
|annual_income        |-0.042214      | 1.000000      |-0.089304         | 0.119274 | ...  |-0.144553 |-0.133055        | 0.042194          |-0.144129 |
|eating_habits        | 0.066318      |-0.089304      | 1.000000         |-0.063661 | ...  |-0.033087 | 0.061451        |-0.109776          | 0.083808|
|addiction_friend     | 0.023710      | 0.119274      |-0.063661         | 1.000000 | ...  |-0.075882 | 0.100138        |-0.034929          |-0.091548|
|addiction            | 0.032259      | 0.055304      | 0.068245         | 0.428328 | ...  | 0.040946 | 0.097530        |-0.078782          |-0.032259|
|medical_history      | 0.175749      |-0.264076      | 0.106260         |-0.093513 | ...  | 0.280714 | 0.412722        |-0.190702          | 0.266967|
|depressed            | 0.053648      |-0.144553      |-0.033087         |-0.075882 | ...  | 1.000000 | 0.285102        |-0.498167          | 0.297989|
|anxiety              | 0.162366      |-0.133055      | 0.061451         | 0.100138 | ...  | 0.285102 | 1.000000        |-0.245867          | 0.377108|
|happy_currently      |-0.050309      | 0.042194      |-0.109776         |-0.034929 | ...  |-0.498167 |-0.245867        | 1.000000          |-0.408260|
|suicidal_thoughts    | 0.080420      |-0.144129      | 0.083808         |-0.091548 | ...  | 0.297989 | 0.377108        |-0.408260          | 1.000000|

[10 rows x 10 columns]


Tuning the Logistic Regression Model with GridSearchCV

Best parameters:  {'C': 1, 'max_iter': 100, 'multi_class': 'ovr', 'solver': 'newton-cg'}
Best cross-validation score:  77.05882352941177 %
Best estimator:  LogisticRegression(C=1, multi_class='ovr', solver='newton-cg')  

Tuning KNN model with GridSearchCV

Best parameters:  {'algorithm': 'auto', 'leaf_size': 20, 'n_neighbors': 9, 'weights': 'uniform'}
Best cross-validation score:  81.91176470588235 %
Best estimator:  KNeighborsClassifier(leaf_size=20, n_neighbors=9)

Tuning Decision Tree model with GridSearchCV

Best parameters:  {'criterion': 'entropy', 'max_depth': 7, 'min_samples_split': 2, 'random_state': 0}
Best cross-validation score:  75.73529411764707 %
Best estimator:  DecisionTreeClassifier(criterion='entropy', max_depth=7, random_state=0)

Tuning Random Forest model with GridSearchCV

Best parameters:  {'criterion': 'entropy', 'max_depth': 5, 'min_samples_split': 3, 'n_estimators': 20, 'random_state': 0}
Best cross-validation score:  80.66176470588235 %
Best estimator:  RandomForestClassifier(criterion='entropy', max_depth=5, min_samples_split=3,
                       n_estimators=20, random_state=0)

Tuning Boosting model with GridSearchCV

Best parameters:  {'learning_rate': 0.8, 'n_estimators': 30, 'random_state': 0}
Best cross-validation score:  78.30882352941177 %
Best estimator:  AdaBoostClassifier(learning_rate=0.8, n_estimators=30, random_state=0)

Tuning Bagging model with GridSearchCV

Best parameters:  {'bootstrap': False, 'bootstrap_features': True, 'max_samples': 0.4, 'n_estimators': 70, 'random_state': 0}
Best cross-validation score:  80.73529411764706 %
Best estimator:  BaggingClassifier(bootstrap=False, bootstrap_features=True, max_samples=0.4,
                  n_estimators=70, random_state=0)

Tuning the Logistic Regression Model with RandomizedSearchCV

Best parameters:  {'C': 59, 'max_iter': 495, 'multi_class': 'multinomial', 'solver': 'newton-cg'}
Best cross-validation score:  72.13235294117646 %
Best estimator:  LogisticRegression(C=59, max_iter=495, multi_class='multinomial',
                   solver='newton-cg')

Tuning KNN model with RandomizedSearchCV

Best parameters:  {'algorithm': 'ball_tree', 'leaf_size': 98, 'n_neighbors': 14, 'weights': 'distance'}
Best cross-validation score:  75.80882352941177 %
Best estimator:  KNeighborsClassifier(algorithm='ball_tree', leaf_size=98, n_neighbors=14,
                     weights='distance')

Tuning Decision Tree model with RandomizedSearchCV

Best parameters:  {'criterion': 'entropy', 'max_depth': 30, 'min_samples_split': 4, 'random_state': 0}
Best cross-validation score:  73.38235294117646 %
Best estimator:  DecisionTreeClassifier(criterion='entropy', max_depth=30, min_samples_split=4,
                       random_state=0)

Tuning Random Forest model with RandomizedSearchCV

Best parameters:  {'criterion': 'entropy', 'max_depth': 5, 'min_samples_split': 4, 'n_estimators': 67, 'random_state': 0}
Best cross-validation score:  76.98529411764706 %
Best estimator:  RandomForestClassifier(criterion='entropy', max_depth=5, min_samples_split=4,
                       n_estimators=67, random_state=0)

Tuning Boosting model with RandomizedSearchCV

Best parameters:  {'learning_rate': 1, 'n_estimators': 14, 'random_state': 0}
Best cross-validation score:  78.30882352941177 %
Best estimator:  AdaBoostClassifier(learning_rate=1, n_estimators=14, random_state=0)

Tuning Bagging model with RandomizedSearchCV

Best parameters:  {'bootstrap': False, 'bootstrap_features': True, 'max_samples': 0.3, 'n_estimators': 80, 'random_state': 0}
Best cross-validation score:  79.41176470588236 %
Best estimator:  BaggingClassifier(bootstrap=False, bootstrap_features=True, max_samples=0.3,
                  n_estimators=80, random_state=0)
accuracyDict:

{
 "Log_Reg_mod_tuning": 85.71428571428571,
 
 "KNN": 90.47619047619048,
 
 "Decision_Tree": 80.95238095238095,
 
 "Random_Forest": 85.71428571428571,
 
 "AdaBoost": 80.95238095238095,
 
 "Bagging": 90.47619047619048,
 
 "Log_Reg_mod_tuning_rand": 85.71428571428571,
 
 "KNN_rand": 85.71428571428571,
 
 "Decision_Tree_rand": 76.19047619047619,
 
 "Random_Forest_rand": 85.71428571428571,
 
 "AdaBoost_rand": 80.95238095238095,
 
 "Bagging_rand": 85.71428571428571
}
