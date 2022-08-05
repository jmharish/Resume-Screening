import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV
import xgboost as xgb 

xg_clf = xgb.XGBClassifier( max_depth=4,reg_lambda = 3,eta = 0.3,gamma = 10 )  


pkl_file = open("C:\Harish\iGreenData_internship\pickled\X_Train.pickle","rb")
X_Train = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open("C:\Harish\iGreenData_internship\pickled\Y_Train.pickle","rb")
Y_Train = pickle.load(pkl_file)
pkl_file.close()


xg_gs = xg_clf.fit(X_Train,Y_Train)


pkl_file = open("C:\Harish\iGreenData_internship\pickled\XGB_tuned.pickle","wb") 
pickle.dump(xg_gs,pkl_file)
pkl_file.close()




pkl_file = open("C:\Harish\iGreenData_internship\pickled\X_Val.pickle","rb")
x_val = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open("C:\Harish\iGreenData_internship\pickled\Y_Val.pickle","rb")
Y_Train = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open("C:\Harish\iGreenData_internship\pickled\XGB_tuned.pickle","rb")
xg_gs = pickle.load(pkl_file)
pkl_file.close()

print(xg_gs.get_params())
