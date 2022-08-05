from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.linear_model import LogisticRegression
import sklearn.metrics
from sklearn import svm 
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV

"""#the calssifiers along with their default parameters are taken
svc_clf = svm.SVC(kernel='linear',probability=True) # probability is set to True to use predict_proba()
lr_clf = LogisticRegression()
knn_clf = knn()


#differnet values of parameters are chosen
# the socing metric used here is fi_micro avg 
# precision or recall as such can not be used in multiclass classification

param = {'C':[0.0001,0.01,1,10,100,1000]}
svm_gs = GridSearchCV(estimator= svc_clf, param_grid= param ,scoring= 'f1_micro')


param = {'n_neighbors':[1,2,5,10,50] , 'weights':['uniform','distance']}
knn_gs = GridSearchCV(estimator= knn_clf, param_grid= param ,scoring= 'f1_micro')



param = {'C':[0.0001,0.01,1,10,100,1000]}
lr_gs = GridSearchCV(estimator= lr_clf, param_grid= param ,scoring= 'f1_micro')

# GridSearchCV objects are returned 



pkl_file = open("C:\Harish\iGreenData_internship\pickled\X_Train.pickle","rb")
X_Train = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open("C:\Harish\iGreenData_internship\pickled\Y_Train.pickle","rb")
Y_Train = pickle.load(pkl_file)
pkl_file.close()



# Taining using the oversampled dataset
knn_gs = knn_gs.fit(X_Train,Y_Train)

lr_gs = lr_gs.fit(X_Train,Y_Train)

svm_gs = svm_gs.fit(X_Train,Y_Train)


# estimators are the GridSearchCV class with  best parameters set 
# though the estimators algorithm varies( like: knn svm etc) the objects are GridSearchCV instances 


#pickling
pkl_file = open("C:\Harish\iGreenData_internship\pickled\KNN_gs.pickle","wb") 
pickle.dump(knn_gs,pkl_file)
pkl_file.close()

pkl_file = open("C:\Harish\iGreenData_internship\pickled\lr_gs.pickle","wb") 
pickle.dump(lr_gs,pkl_file)
pkl_file.close()

pkl_file = open("C:\Harish\iGreenData_internship\pickled\linSVM_gs.pickle","wb") 
pickle.dump(svm_gs,pkl_file)
pkl_file.close()

print(" pickling done")

svm_clf = svm.SVC(C= 0.01,kernel ='linear',probability=True)
svm_clf.fit(X_Train,Y_Train)

pkl_file = open("C:\Harish\iGreenData_internship\pickled\linSVM_tuned.pickle","wb") 
pickle.dump(svm_clf,pkl_file)
pkl_file.close()
"""
pkl_file = open("C:\Harish\iGreenData_internship\pickled\X_Train.pickle","rb")
X_Train = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open("C:\Harish\iGreenData_internship\pickled\Y_Train.pickle","rb")
Y_Train = pickle.load(pkl_file)
pkl_file.close()

lr = LogisticRegression(C = 0.05).fit(X_Train,Y_Train)
pkl_file = open("C:\Harish\iGreenData_internship\pickled\lr_tuned.pickle","wb") 
pickle.dump(lr,pkl_file)
pkl_file.close()