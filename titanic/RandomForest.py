# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

def PredictMissingFeature(targetFeat, otherFeat):
#    targetFeat,otherFeat = feature[:,0].copy(), feature[:,1:].copy()
    nanMask = np.isnan(targetFeat)
    X_train, X_test = otherFeat[1-nanMask], otherFeat[nanMask] 
    y_train, y_test = targetFeat[1-nanMask], targetFeat[nanMask]
    
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train)
    y_test = model.predict(X_test)
    y_test = np.round(y_test).astype(int)
    
    targetFeat[nanMask] = y_test
    
    return np.hstack([targetFeat.reshape([-1,1]), otherFeat])
    
# Path setting
rootPath = "D:/Dataset/kaggle/titanic/"
train_path = rootPath + "train.csv"
test_path = rootPath + "test.csv"
output = rootPath + "output_add_ticket.csv"

# Read data
train_all = pd.read_csv(train_path)
label_all = np.array(train_all["Survived"])
train_all = train_all.drop(["Survived","PassengerId", "Name"], 1)

test = pd.read_csv(test_path)
testID = test["PassengerId"]
test = test.drop(["PassengerId", "Name"], 1)

num_train = len(train_all)

train_test = train_all.append(test)

# Preprocess features
feature_cate = train_test[["Pclass", "Sex", "Embarked", "Ticket"]]
feature_num = train_test[["Age", "SibSp", "Parch", "Fare"]]
feature_cate = pd.get_dummies(feature_cate)
feature = np.hstack([np.array(feature_num),np.array(feature_cate)])
feature = PredictMissingFeature(feature[:,0].copy(), np.delete(feature, 0, axis=1).copy())
feature = PredictMissingFeature(feature[:,3].copy(), np.delete(feature, 3, axis=1).copy())

feature_train = feature[:num_train]
feature_test = feature[num_train:]

#kf = KFold(n_splits=5)
#kf.get_n_splits(train_all)
#
#
#for train_index, val_index in kf.split(train_all):
#    
#    train_data, val_data = feature[train_index],feature[val_index]
#    train_label, val_label = label_all[train_index],label_all[val_index]    
#    
#    RF = RandomForestClassifier(max_depth=10, n_estimators=100)
#    RF.fit(train_data, train_label)
#    prediction = RF.predict(val_data)
#    
#    accuracy = np.sum(prediction==val_label)/len(val_label)
#    print(accuracy)
    


# Tune parameter
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 100, num = 10)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(
        estimator = rf, param_distributions = random_grid, 
        n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(feature_train, label_all)
best_params = rf_random.best_params_

bast_model = RandomForestClassifier(
                n_estimators = best_params["n_estimators"],
                min_samples_split = best_params["min_samples_split"],
                min_samples_leaf = best_params["min_samples_leaf"],
                max_features = best_params["max_features"],
                max_depth = best_params["max_depth"],
                bootstrap = best_params["bootstrap"],
                )

bast_model.fit(feature_train, label_all)
predictions = bast_model.predict(feature_test)
 
submission = pd.DataFrame(testID)
submission["Survived"] = predictions
submission.to_csv(output, index=False)
