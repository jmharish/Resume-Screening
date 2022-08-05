import os
import pandas as pd
from pyresparser import ResumeParser
import pickle
from fpdf import FPDF
pdf = FPDF()
"""
os.chdir("C:\Harish\iGreenData_internship\Kaggle Dataset")
df1 = pd.read_csv("UpdatedResumeDataSet.csv") 
L_index = []
L_ser =[]
L_labels = list(df1["Category"])


for i in range(len(df1["Resume"])):  
    fid = open("Temp1.txt","w+")
    fid.truncate(0)
    fid.close()



    fid = open("Temp1.txt","w+")
    fid.write(df1["Resume"][i].encode('utf8').decode('ascii', 'ignore')) #writing the text of the resume onto a text file
    fid.close()


    pdf = FPDF()   #creating a pdf file adding page and setting font for writing the text into the pdf 
    pdf.add_page()
    pdf.set_font("Arial", size = 5)
    f = open("Temp1.txt","r")
    
    for x in f.readlines():
        pdf.cell(0,5, txt = x.encode('windows-1252').decode('ascii', 'ignore'), ln = 1, align = 'L')
    f.close()
    pdf.output("Temp.pdf")

    di =  ResumeParser("C:\Harish\iGreenData_internship\Kaggle Dataset\Temp.pdf").get_extracted_data() #parsing pdf to pyreparser
    d = dict({})
    for ic in di['skills']:
        d[ic.lower()] = True
    s = pd.Series(d)
    L_ser.append(s)
    L_index.append(L_labels[i] + str(i))
    
    
df = pd.DataFrame(L_ser, index=L_index)
df["Label"] = L_labels



pkl_file = open("C:\Harish\iGreenData_internship\pickled\Kaggle_labeled_dataframe.pickle","wb") #pickling; need not fit every time
pickle.dump(df,pkl_file)
pkl_file.close()
"""
pkl_file = open("C:\Harish\iGreenData_internship\pickled\Kaggle_labeled_dataframe.pickle","rb")
df = pickle.load(pkl_file)
pkl_file.close()
d = pd.DataFrame()

cf = df

cf.drop(cf[cf["Label"]=="HR"].index , inplace = True)
cf.drop(cf[cf["Label"]=="Arts"].index , inplace = True)
cf.drop(cf[cf["Label"]=="Advocate"].index , inplace = True)
cf.drop(cf[cf["Label"]=="Sales"].index , inplace = True)
cf.drop(cf[cf["Label"]=="Health and fitness"].index , inplace = True)
cf.drop(cf[cf["Label"]=="PMO"].index , inplace = True)

# Java Developer to Java
c= cf[cf["Label"]=="Java Developer"]
c.drop(columns = "Label",inplace = True)
c["Label"] = "Java"
cf.drop(cf[cf["Label"]=="Java Developer"].index , inplace = True)
cf = cf.append(c,ignore_index = False )

#Data Science to Data_Analyst
c= cf[cf["Label"]=="Data Science"]
c.drop(columns = "Label",inplace = True)
c["Label"] = "Data_Analyst"
cf.drop(cf[cf["Label"]=="Data Science"].index , inplace = True)
cf = cf.append(c,ignore_index = False )

#Devops Engineer to Dev_Ops
c= cf[cf["Label"]=="DevOps Engineer"]
c.drop(columns = "Label",inplace = True)
c["Label"] = "DevOps"
cf.drop(cf[cf["Label"]=="DevOps Engineer"].index , inplace = True)
cf = cf.append(c,ignore_index = False )




pkl_file = open("C:\Harish\iGreenData_internship\pickled\labeled_dataframe.pickle","rb")
dfb = pickle.load(pkl_file)
pkl_file.close()

print(set(dfb["Label"]))

x = len(dfb["Label"])

pkl_file = open("C:\Harish\iGreenData_internship\pickled\DataScience_dataframe.pickle","rb")
dfd = pickle.load(pkl_file)
pkl_file.close()


print(set(dfd["Label"]))


 
#d = pd.DataFrame().join()
dfb = dfb.append(dfd,ignore_index = False)
dfb = dfb.append(cf,ignore_index = False)
dfb.fillna(False,inplace = True)

pkl_file = open("C:\Harish\iGreenData_internship\pickled\Final_labeled_dataframe.pickle","wb") #pickling; need not fit every time
pickle.dump(dfb,pkl_file)
pkl_file.close()

print(dfb.head())
print(set(dfb["Label"]))