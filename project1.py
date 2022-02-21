###################################
#####    Open ML Evaluation   #####
###################################

# data obtained:
# https://www.openml.org/d/847

# supporting information in the article:
# Haslett, J. and Raftery, A. E. (1989). Space-time Modelling with
# Long-memory Dependence: Assessing Ireland's Wind Power Resource
# (with Discussion). Applied Statistics 38, 1-50.

from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

wind = datasets.fetch_openml(data_id=847)

print("\nThis program takes daily wind speeds from 1961 - 1978 \
from Ireland and applies ML to identify patterns")

print("\nUsing numeric features a decision tree is built to produce nominal targets \
from numeric features") 

# Create the object for the decision tree that will use the Gini Index for entropy
from sklearn import tree
mytree = tree.DecisionTreeClassifier(criterion="entropy")

# fit the tree to the features and target
train = mytree.fit(wind.data,wind.target)
print(tree.export_text(mytree))

# obtain analysis of the model
predictions = mytree.predict(wind.data)
accuracy = metrics.accuracy_score(wind.target, predictions)
precision = metrics.precision_score(wind.target, predictions, pos_label='P')
f1 = metrics.f1_score(wind.target, predictions, pos_label='P')
recall = metrics.recall_score(wind.target, predictions, pos_label='P')

# calculate area under curve (auc) from ROC curve (only works for numeric features)
pp = mytree.predict_proba(wind.data)
auc = metrics.roc_auc_score(wind.target, pp[:,1])

#print("The accuracy of the model is: ",accuracy)
#print("The precision of the model is: ",precision)
#print("The recall of the model is: ",recall)
#print("The F1 score of the model is: ",f1)
#print("The area under the curve from the ROC curve is: ",auc)

# create a decision tree evaluation with default parameters
dtc = tree.DecisionTreeClassifier()
cv = model_selection.cross_validate(dtc, wind.data, wind.target, scoring=["accuracy","roc_auc"], cv=10)
avg_auc = cv["test_roc_auc"].mean()
print("The average AUC of the ROC curve for the default parameter decision tree is: ", avg_auc)

# create a decision tree evaluation with tuned parameters
parameters = [{"min_samples_leaf":[2,4,6,8,10]}]
tuned_dtc = model_selection.GridSearchCV(dtc, parameters, scoring="roc_auc", cv = 5)
cv_tuned = model_selection.cross_validate(tuned_dtc, wind.data, wind.target, scoring=["accuracy","roc_auc"], cv=10, return_train_score=True)
avg_auc_tuned = cv_tuned["test_roc_auc"].mean()
print("The average AUC of the ROC curve for the tuned decision tree is: ",avg_auc_tuned)

# create a random forest evaluation
rf = RandomForestClassifier()
cv_rf = model_selection.cross_validate(rf, wind.data, wind.target, scoring=["accuracy", "roc_auc"], cv=10)
avg_auc_rf = cv_rf["test_roc_auc"].mean()
print("The average AUC of the ROC for the random forest is: ",avg_auc_rf)

# create a  bagged decision tree evaluation
bagged_dtc = BaggingClassifier()
cv_bagged = model_selection.cross_validate(bagged_dtc, wind.data, wind.target, scoring=["accuracy", "roc_auc"], cv=10)
avg_auc_bagged = cv_bagged["test_roc_auc"].mean()
print("The average AUC of the ROC for the bagged decision tree is: ", avg_auc_bagged)

# create an AdaBoosted decision tree evaluation
ada_dtc = AdaBoostClassifier()
cv_adaboost = model_selection.cross_validate(ada_dtc, wind.data, wind.target, scoring=["accuracy", "roc_auc"], cv = 10)
avg_auc_adaboost = cv_adaboost["test_roc_auc"].mean()
print("The average AUC of the ROC for the AdaBoosted decision tree is: ",avg_auc_adaboost)

# now figure out which model had the higest AUC
avg_auc_rounded = round(avg_auc, 3)
avg_auc_tuned_rounded = round(avg_auc_tuned, 3)
avg_auc_rf_rounded = round(avg_auc_rf, 3)
avg_auc_bagged_rounded = round(avg_auc_bagged, 3)
avg_auc_adaboost_rounded = round(avg_auc_adaboost, 3)

evaluate = {"Default parameters": avg_auc_rounded,
            "Tuned parameters": avg_auc_tuned_rounded,
            "Random Forest": avg_auc_rf_rounded,
            "Bagged": avg_auc_bagged_rounded,
            "AdaBoost": avg_auc_adaboost_rounded}

sort_evaluate = sorted(evaluate.items(), key=lambda x:x[1], reverse=True)
for x in sort_evaluate:
    print(x[0], x[1])
    





