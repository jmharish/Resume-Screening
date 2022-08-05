from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.linear_model import LogisticRegression
from sklearn import svm 
import xgboost as xgb
from sklearn.ensemble import  VotingClassifier
import pickle

pkl_file = open("C:\Harish\iGreenData_internship\pickled\X_Train.pickle","rb")
X_Train = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open("C:\Harish\iGreenData_internship\pickled\Y_Train.pickle","rb")
Y_Train = pickle.load(pkl_file)
pkl_file.close()


mNB_clf = MultinomialNB(alpha = 100)
bNB_clf = BernoulliNB(alpha = 100)
cNB_clf = ComplementNB(alpha = 100)
svm_clf = svm.SVC(C= 0.01,kernel ='linear',probability=True)
xg_clf = xgb.XGBClassifier( max_depth=4,reg_lambda = 3,eta = 0.3,gamma = 10 ) 
lr_clf = LogisticRegression(C = 0.05)
kn =knn(n_neighbors=2,weights="uniform")

models =[("mnb",mNB_clf), ("bnb",bNB_clf),('cnb',cNB_clf),('svm',svm_clf),('XG',xg_clf),("lr",lr_clf),('KNN',kn)]
vote_clf = VotingClassifier(estimators=models,voting='soft')
vote_clf = vote_clf.fit(X_Train,Y_Train)

pkl_file = open("C:\Harish\iGreenData_internship\pickled\Voted_clf.pickle","wb") 
pickle.dump(vote_clf,pkl_file)
pkl_file.close()