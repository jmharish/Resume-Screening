
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.linear_model import LogisticRegression
from sklearn import svm 
import pickle

pkl_file = open("C:\Harish\iGreenData_internship\pickled\X_Train.pickle","rb")
X_Train = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open("C:\Harish\iGreenData_internship\pickled\Y_Train.pickle","rb")
Y_Train = pickle.load(pkl_file)
pkl_file.close()

X_Train.reset_index(inplace=True , drop= True)
Y_Train.reset_index(inplace=True , drop= True)

mNB_clf = MultinomialNB().fit(X_Train,Y_Train)
bNB_clf = BernoulliNB().fit(X_Train,Y_Train)
cNB_clf = ComplementNB().fit(X_Train,Y_Train)

pkl_file = open("C:\Harish\iGreenData_internship\pickled\MulNB.pickle","wb") 
pickle.dump(mNB_clf,pkl_file)
pkl_file.close()

pkl_file = open("C:\Harish\iGreenData_internship\pickled\BerNB.pickle","wb") 
pickle.dump(bNB_clf,pkl_file)
pkl_file.close()

pkl_file = open("C:\Harish\iGreenData_internship\pickled\ComNB.pickle","wb") 
pickle.dump(cNB_clf,pkl_file)
pkl_file.close()

knn_clf = knn().fit(X_Train,Y_Train)
pkl_file = open("C:\Harish\iGreenData_internship\pickled\KNN.pickle","wb") 
pickle.dump(knn_clf,pkl_file)
pkl_file.close()

linSVM_clf = svm.SVC(kernel='linear').fit(X_Train,Y_Train)
pkl_file = open("C:\Harish\iGreenData_internship\pickled\linSVM.pickle","wb") 
pickle.dump(linSVM_clf,pkl_file)
pkl_file.close()

lr = LogisticRegression().fit(X_Train,Y_Train)
pkl_file = open("C:\Harish\iGreenData_internship\pickled\lr.pickle","wb") 
pickle.dump(lr,pkl_file)
pkl_file.close()
