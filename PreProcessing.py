import os 
import pandas as pd
from pyresparser import ResumeParser
import pickle

os.chdir("C:\Harish\iGreenData_internship\DataScience_resumes")
l = os.listdir("C:\Harish\iGreenData_internship\DataScience_resumes")
L_ser =[]
L_labels = []
L_index = []
for i in l:
    di =  ResumeParser(i).get_extracted_data()
    d = dict({})
    for ic in di['skills']:
        d[ic.lower()] = True
    s = pd.Series(d)
    L_ser.append(s)
    L_index.append(i)
    L_labels.append("Data_Analyst")

df2 = pd.DataFrame(L_ser,index = L_index)








df2["Label"] = L_labels

pkl_file = open("C:\Harish\iGreenData_internship\pickled\DataScience_dataframe.pickle","wb") #pickling; need not fit every time
pickle.dump(df2,pkl_file)
pkl_file.close()

pkl_file = open("C:\Harish\iGreenData_internship\pickled\DataScience_dataframe.pickle","rb")
df2 = pickle.load(pkl_file)
pkl_file.close()

print(df2.head())

    



