from pandas import DataFrame
import pandas as pd 
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import matplotlib.pyplot as plt

pkl_file = open("C:\Harish\iGreenData_internship\pickled\Final_labeled_dataframe.pickle","rb")
df = pickle.load(pkl_file)
pkl_file.close()







print(set(df["Label"]))

Y = df["Label"]
X = df.drop(columns = ["Label"])
"""print( "number of labels:######",len(list(set(df["Label"]))))
skf5 = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
skf4 = StratifiedKFold(n_splits=4, random_state=42, shuffle=True)
for train_index, test_index in skf5.split(X, Y):  # test set gets 20% of dataset (1/5)
    X_train_val, x_test = X.iloc[train_index], X.iloc[test_index]
    Y_train_val, y_test = Y.iloc[train_index], Y.iloc[test_index]

for train_index, test_index in skf4.split(X_train_val , Y_train_val ): # val. set gets 25%(1/4) of 80%(->X_train_val) of dataset i.e 20% of dataset
    X_train, x_val = X.iloc[train_index], X.iloc[test_index]
    Y_train, y_val = Y.iloc[train_index], Y.iloc[test_index]



print(" train lenghts are:",len(X_train)," " ,len(Y_train))
print(X_train[0:5])
print(Y_train[0:5])







print(" testing lenghts are:",len(x_test)," " ,len(y_test))
print(x_test[0:5])
print(y_test[0:5])

print(" validation lenghts are:",len(x_val)," " ,len(y_val))
print(x_val[0:5])
print(y_val[0:5])

plt.figure()
plt.subplot(3,1,1)
c = Counter(Y_train)
d = dict(c)
x_axis = list(set(df["Label"]))
y_axis = [d[i] for i in x_axis]

plt.scatter(x_axis,y_axis)
t = plt.xticks([])
plt.title("Train Set")

plt.subplot(3,1,2)
c = Counter(y_val)
d = dict(c)
x_axis = list(set(df["Label"]))
y_axis = [d[i] for i in x_axis]

plt.scatter(x_axis,y_axis)
t = plt.xticks([]) #removes labels in the x axis
plt.title("Validation Set")

plt.subplot(3,1,3)
c = Counter(y_val)
d = dict(c)
x_axis = list(set(df["Label"]))
y_axis = [d[i] for i in x_axis]

plt.scatter(x_axis,y_axis)
t = plt.xticks(rotation= 90) #rotates the labels in x axis by 90 degrees
plt.title("Test Set")

plt.show()
"""
print(df.head())

"""
#pickling the train sets 
pkl_file = open("C:\Harish\iGreenData_internship\pickled\X_Train.pickle","wb") 
pickle.dump(X_train,pkl_file)
pkl_file.close()

pkl_file = open("C:\Harish\iGreenData_internship\pickled\Y_Train.pickle","wb") 
pickle.dump(Y_train,pkl_file)
pkl_file.close()

# pickling the validation sets 
pkl_file = open("C:\Harish\iGreenData_internship\pickled\X_Val.pickle","wb") 
pickle.dump(x_val,pkl_file)
pkl_file.close()

pkl_file = open("C:\Harish\iGreenData_internship\pickled\Y_Val.pickle","wb") 
pickle.dump(y_val,pkl_file)
pkl_file.close()

#pickling test sets

pkl_file = open("C:\Harish\iGreenData_internship\pickled\X_Test.pickle","wb") 
pickle.dump(x_test,pkl_file)
pkl_file.close()

pkl_file = open("C:\Harish\iGreenData_internship\pickled\Y_Test.pickle","wb") 
pickle.dump(y_test,pkl_file)
pkl_file.close()

print("pickling done")
"""