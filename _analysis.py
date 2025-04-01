import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

fin.describe(include=[‘object’])
fin.groupby (‘Gender’)[‘Gold’].sum().reset_index()
fin[‘Objective’].value_counts()
result=fin[(fin[‘Purpose’]==‘Wealth Creation’) & (fin[‘Gold’]>5)]
percentage=(len(result)/len(fin) * 100
print(f”the query result is {percentage:2f} % of the top wealth creators.”)
demo=fin.groupby([‘Gender’,’age’])[‘Avenue’].value_counts()
def whale (row):
     if row [‘mutual_Funds’] <= 2 or row [‘Gold’] <= 2: 
        return ‘lightweight’
     elif row [‘Mutual_Funds’] <= 4 or  row[‘Gold’] <= 3: 
         return ‘Heavyweight’
fin[‘category’] =fin.apply(whale,axis=1)
fin[fin[‘category’] ==‘Middleweight’]

            Machine learning 
df=finance[[‘age’,’Investment _Avenues’,’Objective’,’Avenue’]]
df
X=of.get_dummies(X,columns=[‘Investment_Avenues’,’Objective’],drop_first=True)
X[‘Investment_Avenues_Yes]]=X[‘Investment_Avenues_Yes].astype(int)
X[‘Objective_Growth’]=X[‘Objective_Growth’].astype(int)
X[‘Objective_Income’]=X[‘Objective_Income’].astype(int) 
X
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score,classification_report 
#split data into training and testing 
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state = 42)
#Create a Random Forest Classifier 
rf = RandomForestClassifier(n_estimators =100,random_state=42)
#Train the model 
rf.fit(X_train,y_train)
#make predictions 
y_pred=rf.predict(X_test)
y_pred = rf.predict([[50,1,0,0]])
y_pred 
