import numpy as np
import sklearn as sk
from sklearn import linear_model
from sklearn import tree
from sklearn import utils
from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import pydot as pyd
import pandas as pd
import pathlib
from pathlib import Path
import matplotlib.pyplot as plt

P = Path(__file__).parent
rst=42
dataloc = str(P) + "/Healthcare-Diabetes.csv"
frame = pd.read_csv(dataloc)
features = frame.loc[:, 'Pregnancies':'Age']
notremoved = features[features[['Glucose', 'BloodPressure', 'SkinThickness', 'BMI']] != 0]
notremoved = notremoved.dropna(axis=0, subset=['Glucose', 'BloodPressure', 'SkinThickness', 'BMI'])
features = features.loc[notremoved.index, :]
featurestrain = features.sample(frac=0.8, replace=False, random_state=rst)
featurestest = features.drop(featurestrain.index)
X = features
Y = frame.loc[features.index, 'Outcome']
xtrain = featurestrain
xtest = featurestest
ytrain = frame.loc[featurestrain.index, 'Outcome']
ytest = frame.loc[featurestest.index, 'Outcome']


logmodel = linear_model.LogisticRegression(max_iter=300, random_state=rst).fit(xtrain, ytrain)
acclog = logmodel.score(xtest, ytest)
trainacclog = logmodel.score(xtrain, ytrain)
print("Train log accuracy", trainacclog)
print("Test logistic accuracy", acclog)

permodel = linear_model.Perceptron(max_iter=300,random_state=rst).fit(xtrain, ytrain)
accper = permodel.score(xtest, ytest)
trainaccper = permodel.score(xtrain, ytrain)
print("Train perceptron accuracy", trainaccper)
print("Test perceptron accuracy", accper)

treemodel = tree.DecisionTreeClassifier(random_state=rst).fit(xtrain, ytrain)
acctree = treemodel.score(xtest, ytest)
trainacctree = treemodel.score(xtrain, ytrain)
treedepth = treemodel.get_depth()
treeleaves = treemodel.get_n_leaves()
print("Train tree acc", trainacctree)
print("Test tree accuracy", acctree)
print("Tree depth", treedepth)
print("Tree leaves", treeleaves)

prtreemodel = tree.DecisionTreeClassifier(max_depth=5, max_leaf_nodes=20, random_state=rst).fit(xtrain, ytrain)
pracctree = prtreemodel.score(xtest, ytest)
prtrainacctree = prtreemodel.score(xtrain, ytrain)
prtreedepth = prtreemodel.get_depth()
prtreeleaves = prtreemodel.get_n_leaves()
print("Train reduced tree acc", prtrainacctree)
print("Test reduced tree accuracy", pracctree)
print("Reduced tree depth", prtreedepth)
print("Reduced tree leaves", prtreeleaves)



#print(np.size(xtrain.loc[:,'Pregnancies']))

baglog = ensemble.BaggingClassifier(estimator=linear_model.LogisticRegression(max_iter=300), n_estimators=500, random_state=rst)
baglog.fit(xtrain, ytrain)
blsc = baglog.score(xtest, ytest)
blsc2 = baglog.score(xtrain, ytrain)
print("Train bagging logistic accuracy", blsc2)
print("Test bagging logistic accuracy", blsc)

bagper = ensemble.BaggingClassifier(estimator=linear_model.Perceptron(max_iter=300), n_estimators=500, max_samples=1, random_state=rst).fit(xtrain, ytrain)
bpsc = bagper.score(xtest, ytest)
bpsc2 = bagper.score(xtrain, ytrain)
print("Train bagging perceptron accuracy", bpsc2)
print("Test bagging perceptron accuracy", bpsc)

formod = ensemble.RandomForestClassifier(max_depth=5, max_leaf_nodes=20, random_state=rst).fit(xtrain, ytrain)
foracc = formod.score(xtest, ytest)
foracctrain = formod.score(xtrain, ytrain)
print("Forest train accuracy", foracctrain)
print("Forest test accuracy", foracc)


#rf = RandomForestClassifier(n_estimators=100, max_depth=5, max_leaf_nodes=20)
#rf.fit(xtrain, ytrain)
#fn=list(features.columns)
#cn=["Absent","Present"]
#fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (16,16), dpi=1200)
#tree.plot_tree(rf.estimators_[0],
               #feature_names = fn,
               #class_names=cn,
               #filled = True, label='none', impurity=False, fontsize=6, ax=axes)
#fig.savefig('rf_individualtree.png')


n_estimators = [30, 50, 100, 200, 400, 500, 1000]
max_features = ['sqrt']
max_features.append(None)
#max_depth = [int(x) for x in np.linspace(5, 20, num = 10)]
#max_depth = [int(x) for x in np.linspace(5, 10, num = 5)]
max_depth = [5]
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
#max_leaf_nodes = [int(x) for x in np.linspace(2, 40, num = 10)]
max_leaf_nodes = [20]
random_grid = {'n_estimators': n_estimators, 'max_depth':max_depth, 'max_features': max_features, 'min_samples_split':min_samples_split, 'min_samples_leaf':min_samples_leaf, 'max_leaf_nodes':max_leaf_nodes}
print(random_grid)


rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 20, cv = 15, verbose=1, random_state=rst, n_jobs = -1)
rf_random.fit(xtrain, ytrain)

print(rf_random.best_params_)
print(rf_random.best_estimator_)
randoacctrain = rf_random.score(xtrain, ytrain)
randoacctest = rf_random.score(xtest, ytest)
print("Rando training accuracy", randoacctrain)
print("Rando testing accuracy", randoacctest)

#{'n_estimators': 30, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_leaf_nodes': 60, 'max_features': None, 'max_depth': 18}
#Rando training accuracy 0.9896507115135834
#Rando testing accuracy 0.9559585492227979

#{'n_estimators': 30, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_leaf_nodes': 60, 'max_features': None, 'max_depth': 18}
#Rando training accuracy 0.9883570504527813
#Rando testing accuracy 0.9533678756476683

#{'n_estimators': 30, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_leaf_nodes': 60, 'max_features': None, 'max_depth': 18}
#Rando training accuracy 0.9890038809831824
#Rando testing accuracy 0.9818652849740933

param_grid = {
    'max_depth':[5],
    'max_features':['sqrt', None],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split':[2, 5],
    'n_estimators':[30, 200, 500, 1000],
    'max_leaf_nodes':[20]
}

rf_grid = GridSearchCV(estimator = rf, param_grid=param_grid, cv = 15, verbose = 1, n_jobs = -1)
rf_grid.fit(xtrain, ytrain)

print(rf_grid.best_params_)
gridacctrain = rf_grid.score(xtrain, ytrain)
gridacctest = rf_grid.score(xtest, ytest)
print("Grid train accuracy", gridacctrain)
print("Grid test accuracy", gridacctest)

rftree = rf_random.best_estimator_.estimators_[0]
export_graphviz(rftree, out_file = 'rftree.dot', feature_names = list(xtrain.columns), class_names=["Absent","Present"], rounded=True, precision = 1)
(graph, ) = pyd.graph_from_dot_file('rftree.dot')
graph.write_png('rftree.png')

gridtree = rf_grid.best_estimator_.estimators_[0]
export_graphviz(gridtree, out_file = 'gridtree.dot', feature_names = list(xtrain.columns), class_names=["Absent", "Present"], rounded=True, precision = 1)
(graph2, ) = pyd.graph_from_dot_file('gridtree.dot')
graph2.write_png('gridtree.png')







#Fitting 10 folds for each of 30 candidates, totalling 300 fits
#{'n_estimators': 500, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_leaf_nodes': 20, 'max_features': None, 'max_depth': 5}
#RandomForestClassifier(max_depth=5, max_features=None, max_leaf_nodes=20,
#                       n_estimators=500)
#Rando training accuracy 0.9042690815006468
#Rando testing accuracy 0.8471502590673575
#Fitting 10 folds for each of 8 candidates, totalling 80 fits
#{'max_depth': 5, 'max_features': None, 'max_leaf_nodes': 20, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 1000}
#Grid train accuracy 0.9023285899094438
#Grid test accuracy 0.8523316062176166


#Forest train accuracy 0.8861578266494179
#Forest test accuracy 0.8730569948186528
#{'n_estimators': [30, 50, 100, 200, 400, 500, 1000], 'max_depth': [5], 'max_features': ['sqrt', None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'max_leaf_nodes': [20]}
#Fitting 10 folds for each of 30 candidates, totalling 300 fits
#{'n_estimators': 500, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_leaf_nodes': 20, 'max_features': None, 'max_depth': 5}
#RandomForestClassifier(max_depth=5, max_features=None, max_leaf_nodes=20,
#                       n_estimators=500)
#Rando training accuracy 0.9100905562742562
#Rando testing accuracy 0.8808290155440415
#Fitting 10 folds for each of 16 candidates, totalling 160 fits
#{'max_depth': 5, 'max_features': None, 'max_leaf_nodes': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 30}
#Grid train accuracy 0.8990944372574385
#Grid test accuracy 0.8704663212435233

#Forest train accuracy 0.8758085381630013
#orest test accuracy 0.8549222797927462
#{'n_estimators': [30, 50, 100, 200, 400, 500, 1000], 'max_depth': [5], 'max_features': ['sqrt', None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'max_leaf_nodes': [20]}
#Fitting 10 folds for each of 30 candidates, totalling 300 fits
#{'n_estimators': 500, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_leaf_nodes': 20, 'max_features': None, 'max_depth': 5}
#RandomForestClassifier(max_depth=5, max_features=None, max_leaf_nodes=20,
#                       min_samples_leaf=4, min_samples_split=5,
#                       n_estimators=500)
#Rando training accuracy 0.9055627425614489
#Rando testing accuracy 0.8989637305699482
#Fitting 10 folds for each of 16 candidates, totalling 160 fits
#{'max_depth': 5, 'max_features': None, 'max_leaf_nodes': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 500}
#Grid train accuracy 0.9036222509702458
#Grid test accuracy 0.8782383419689119

#Forest train accuracy 0.8790426908150065
#Forest test accuracy 0.8134715025906736
#{'n_estimators': [30, 50, 100, 200, 400, 500, 1000], 'max_depth': [5], 'max_features': ['sqrt', None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'max_leaf_nodes': [20]}
#Fitting 15 folds for each of 20 candidates, totalling 300 fits
#{'n_estimators': 200, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_leaf_nodes': 20, 'max_features': None, 'max_depth': 5}
#RandomForestClassifier(max_depth=5, max_features=None, max_leaf_nodes=20,
#                       min_samples_leaf=2, min_samples_split=5,
#                       n_estimators=200)
#Rando training accuracy 0.9055627425614489
#Rando testing accuracy 0.8341968911917098
#Fitting 15 folds for each of 48 candidates, totalling 720 fits
#{'max_depth': 5, 'max_features': None, 'max_leaf_nodes': 20, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 30}
#Grid train accuracy 0.907503234152652
#Grid test accuracy 0.8419689119170984

#Forest train accuracy 0.8783958602846055
#Forest test accuracy 0.8290155440414507
#{'n_estimators': [30, 50, 100, 200, 400, 500, 1000], 'max_depth': [5], 'max_features': ['sqrt', None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'max_leaf_nodes': [20]}
#Fitting 15 folds for each of 20 candidates, totalling 300 fits
#{'n_estimators': 500, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_leaf_nodes': 20, 'max_features': None, 'max_depth': 5}
#RandomForestClassifier(max_depth=5, max_features=None, max_leaf_nodes=20,
#                       min_samples_leaf=4, min_samples_split=5,
#                       n_estimators=500)
#Rando training accuracy 0.9100905562742562
#Rando testing accuracy 0.8601036269430051
#Fitting 15 folds for each of 48 candidates, totalling 720 fits
#{'max_depth': 5, 'max_features': None, 'max_leaf_nodes': 20, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 30}
#Grid train accuracy 0.8984476067270375
#Grid test accuracy 0.8601036269430051