from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
import sklearn.metrics
from sklearn import svm 
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV

pkl_file = open("C:\Harish\iGreenData_internship\pickled\lr_gs.pickle","rb")
lr = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open("C:\Harish\iGreenData_internship\pickled\KNN_tuned.pickle","rb")
kn = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open("C:\Harish\iGreenData_internship\pickled\linSVM_tuned.pickle","rb")
svc = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open("C:\Harish\iGreenData_internship\pickled\ComNB.pickle","rb")
comNB = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open("C:\Harish\iGreenData_internship\pickled\BerNB.pickle","rb")
bNB = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open("C:\Harish\iGreenData_internship\pickled\MulNB.pickle","rb")
mNB = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open("C:\Harish\iGreenData_internship\pickled\XGB_tuned.pickle","rb")
xg = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open("C:\Harish\iGreenData_internship\pickled\Voted_clf.pickle","rb")
vote_clf = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open("C:\Harish\iGreenData_internship\pickled\X_Test.pickle","rb")
x_test = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open("C:\Harish\iGreenData_internship\pickled\Y_Test.pickle","rb")
y_test = pickle.load(pkl_file)
pkl_file.close()


print(type(x_test))
def test(X,names):
    l_prob = []
    l_prob = vote_clf.predict_proba(X)
    df = pd.DataFrame()
    df["Name"] = list(X.index)
    l = ['Automation Testing', 'Blockchain', 'Business Analyst', 'Civil Engineer', 'Data_Analyst', 'Database', 'DevOps', 'DotNet Developer', 'ETL Developer', 'Electrical Engineering', 'Hadoop', 'Java', 'Mechanical Engineer', 'Network Security Engineer', 'Operations Manager', 'Python Developer', 'SAP Developer', 'SRE', 'Testing', 'Web Designing']
    for j in range(len(l)) :
        df[l[j]] = [i[j] for i in l_prob]
    l_lbl = list(vote_clf.predict(X))
    ma_prob =0
    mi_prob=0
    avg_prob =0
    k =[]
    for i in l_prob:
        #l_lbl.append(l[i.index(max(i))])
        k.append(max(i))
    ma_prob = max(k)
    mi_prob = min(k)
    avg_prob = sum(k)/len(k)
    df["Predicted Label"] = l_lbl 
    df["Probablity of the Predicted Label"] = k
    fid = open("C:\Harish\iGreenData_internship\pickled\Output.csv","w")
    df.to_csv(fid)  #stored in the csv file
    fid.close()
    print(df)
    print("MAXIMUM PREDICTION PROBABLITY OF TRUE LABEL:",ma_prob)
    print("MINIMUM PREDICTION PROBABLITY OF TRUE LABEL:",mi_prob)
    print("AVERAGE PREDICTION PROBABLITY OF TRUE LABEL:",avg_prob)

test(x_test,y_test)
#df = pd.DataFrame().reset_index()


"""    l1 = lr.predict_proba(X)  #-> shape: samples x classes 
    l2 = kn.predict_proba(X)
    l3 = svc.predict_proba(X)
    l4 = mNB.predict_proba(X)
    l5 = comNB.predict_proba(X)
    l6 = bNB.predict_proba(X)
    l7 = xg.predict_proba(X)
    # gets the probablity of each sample belonging to each of the classes
    l_prob = []
    for i in range(len(l1)):
        a =[]
        for j in range(len(l1[0])):
            x = l1[i][j] +  l2[i][j] + l3[i][j] + l4[i][j] + l5[i][j] + l6[i][j] + l7[i][j]
            a.append(x/7) #all the probablies are added and average is taken
        l_prob.append(a)
"""