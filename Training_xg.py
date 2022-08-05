import xgboost as xgb 
import pickle

"""xg_clf = xgb.XGBClassifier()
pkl_file = open("C:\Harish\iGreenData_internship\pickled\X_Train.pickle","rb")
X_Train = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open("C:\Harish\iGreenData_internship\pickled\Y_Train.pickle","rb")
Y_Train = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open("C:\Harish\iGreenData_internship\pickled\X_Val.pickle","rb")
x_val = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open("C:\Harish\iGreenData_internship\pickled\Y_Val.pickle","rb")
y_val = pickle.load(pkl_file)
pkl_file.close()



xg_clf.fit(X_Train,Y_Train,eval_metric="auc")

pkl_file = open("C:\Harish\iGreenData_internship\pickled\XGB.pickle","wb") 
pickle.dump(xg_clf,pkl_file)
pkl_file.close()"""

pkl_file = open("C:\Harish\iGreenData_internship\pickled\XGB.pickle","rb")
xg_clf = pickle.load(pkl_file)
pkl_file.close()
print(xg_clf.get_params())
#booster : gbtree , dart tree type boosters 

#xgb.XGBClassifier().get_params()