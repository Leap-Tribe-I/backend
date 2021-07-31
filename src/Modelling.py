# #Logistic Regression Model
# def log_reg_mod():
#     #training the data in Log reg model
#     lr = LogisticRegression()
#     lr.fit(X_train,y_train)
#     #Predicting the data
#     y_pred_class = lr.predict(X_test)
#     accuracy = evalModel(lr, y_test, y_pred_class)
#     accuracyDict['Log_Reg'] = accuracy * 100
# log_reg_mod()

# #knn Model
# def knn():
#     knn = KNeighborsClassifier(n_neighbors=15)
#     knn.fit(X_train,y_train)
#     y_pred_class = knn.predict(X_test)
#     accuracy = evalModel(knn, y_test, y_pred_class)
#     accuracyDict['KNN'] = accuracy * 100
# knn()

# #Decision Tree Model
# def disTree():
#     dt = DecisionTreeClassifier(criterion='entropy')
#     dt.fit(X_train,y_train)
#     y_pred_class = dt.predict(X_test)
#     accuracy = evalModel(dt, y_test, y_pred_class)
#     accuracyDict['Decision Tree'] = accuracy * 100
# disTree()

# #Random Forest Model
# def randFor():
#     rf = RandomForestClassifier(n_estimators=20, random_state=1)
#     rf.fit(X_train,y_train)
#     y_pred_class = rf.predict(X_test)
#     accuracy = evalModel(rf, y_test, y_pred_class)
#     accuracyDict['Random Forest'] = accuracy * 100
# randFor()