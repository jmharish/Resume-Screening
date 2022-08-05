from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
import sklearn.metrics
from sklearn import svm 
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np
MultinomialNB()
BernoulliNB()
ComplementNB()
LogisticRegression()#C
svm.SVC()#c   #linear fixed
knn()#weight: uniform diatance k
def clf_Score(clf):


    pkl_file = open("C:\Harish\iGreenData_internship\pickled\X_Val.pickle","rb")
    x_val = pickle.load(pkl_file)
    pkl_file.close()

    pkl_file = open("C:\Harish\iGreenData_internship\pickled\Y_Val.pickle","rb")
    y_val = pickle.load(pkl_file)
    pkl_file.close()

    x_val.reset_index(inplace=True , drop= True)
    y_val.reset_index(inplace=True , drop= True)


    

    # scores
    y_pred = clf.predict(x_val)

    pre_score = precision_score(y_val,y_pred,average='micro')
    acc_score = sklearn.metrics.accuracy_score(y_val,y_pred)
    rec_score = sklearn.metrics.recall_score(y_val,y_pred,average='micro')
    F1_score = sklearn.metrics.f1_score(y_val,y_pred,average='micro')
    

    print('precision micro average:',pre_score)
    print('Recall micro average :',rec_score)
    print('f1 micro average :',F1_score)

    # REPORT
    di = sklearn.metrics.classification_report(y_val,y_pred)
    print(di)
    
    
    #Precision _ Recall of each class
    l_prob = clf.predict_proba(x_val) #-> samples x classes
    df = pd.DataFrame()
    l = list(clf.classes_)
    for j in range(len(l)) :
        df[l[j]] = [i[j] for i in l_prob]
    df.set_index(y_val,inplace=True)
    i = 0
    """while(i<20):
        plt.figure()
        for j in range(4):
            plt.subplot(2,2,j+1)
            pr , re  , th= precision_recall_curve(y_val,df[l[i]],pos_label=l[i])
            plt.plot(re,pr)
            tit = "Precision vs Recall:"+l[i]
            if(j+1  < 3):
                plt.xticks([])
            plt.title(tit)
            i = i+1"""
    
    # ROC curve 
    i = 0
    while(i<20):
        plt.figure()
        for j in range(4):
            plt.subplot(2,2,j+1)
            fpr , tpr  , th= roc_curve(y_val,df[l[i]],pos_label=l[i])
            plt.plot(fpr,tpr)
            tit = "ROC curve:"+l[i]
            if(j+1 != 4):
                plt.xticks([])
            plt.title(tit)
            i = i+1
    # AUC SCORE 
    macro_auc = roc_auc_score(y_val,l_prob,average='macro',multi_class='ovr')
    print("AUC score macro average",macro_auc)

    plt.show()

pkl_file = open("C:\Harish\iGreenData_internship\pickled\Voted_clf.pickle","rb") 
clf = pickle.load(pkl_file)
pkl_file.close()
clf_Score(clf)
#print(list(clf.classes_))
