import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, datasets
import matplotlib.pyplot as plt
import warnings
import os
from compare_classifiers import comparison_plot
from classifiers import svm_classifier, dtree_classifier, boost_classifier, knn_classifier, neural_net, tuned_boosting_classifier, tuned_neural_net, tuned_dtree_classifier, tuned_svm_classifier, tuned_knn_classifier, tuned_weighted_knn_classifier

warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
os.system('clear')


train_heart = np.array(pd.read_csv('SPECTF.train'))
test_heart = np.array(pd.read_csv('SPECTF.test'))

train_labels = train_heart[:,0]
test_labels = test_heart[:,0]
x = np.vstack((train_heart[:,1:],test_heart[:,1:]))
y = np.hstack((train_labels,test_labels))
print(x.shape,y.shape)

x = preprocessing.scale(x)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=3169)
# Apply different machine learning classifiers

svm_classifier(X_train,y_train,X_test,y_test,'linear','auto',plotting=False)

dtree_classifier(X_train,y_train,X_test,y_test,plotting=False)

boost_classifier(X_train,y_train,X_test,y_test,plotting=False)

knn_classifier(X_train,y_train,X_test,y_test, neighbors=2, plotting=False)

neural_net(X_train,y_train,X_test,y_test,learning_rate=1e-01,plotting=False)

print('********************************')

best_svm = tuned_svm_classifier(X_train,y_train,X_test,y_test,'linear','auto',plotting=False)

best_dtree = tuned_dtree_classifier(X_train,y_train,X_test,y_test,plotting=False)

best_knn = tuned_knn_classifier(X_train,y_train,X_test,y_test,neighbors=2, plotting=False)

best_knn_weighted = tuned_weighted_knn_classifier(X_train,y_train,X_test,y_test,neighbors=2, plotting=False)

best_nn = tuned_neural_net(X_train,y_train,X_test,y_test,learning_rate=0.0001,plotting=False)

best_boost = tuned_boosting_classifier(X_train,y_train,X_test,y_test,plotting=False)

comparison_plot(x,y,best_knn['n_neighbors'],'auto',best_svm['C'],best_dtree['max_depth'],best_nn['alpha'],best_boost['n_estimators'])




