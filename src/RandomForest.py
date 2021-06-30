# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import pickle


datasetPath="./dataset.xls"


# %%
data = pd.read_excel(datasetPath)

# dropping first row
data.drop('Unnamed: 0', axis = 1, inplace = True)

# fill the empty values with the mean value of the column Close_Value
close_value_mean =data['Close_Value'].mean()
data['Close_Value'].fillna(close_value_mean, inplace = True)


# drop all the row which the Stage is 'In Progress'
index_names = data[data['Stage'] == 'In Progress'].index
data.drop(index_names , inplace=True)


# %%
data['Stage'] = data['Stage'].astype('category')
data['Customer'] = data['Customer'].astype('category')
data['Agent'] = data['Agent'].astype('category')
data['SalesAgentEmailID'] = data['SalesAgentEmailID'].astype('category')
data['ContactEmailID'] = data['ContactEmailID'].astype('category')
data['Product'] = data['Product'].astype('category')

# %%
data['Stage']=data['Stage'].cat.codes
data['Customer']=data['Customer'].cat.codes
data['Agent']=data['Agent'].cat.codes
data['SalesAgentEmailID']=data['SalesAgentEmailID'].cat.codes
data['ContactEmailID']=data['ContactEmailID'].cat.codes
data['Product']=data['Product'].cat.codes

# %%
data['Created Date'] = data['Created Date'].astype(int)
data['Close Date'] = data['Close Date'].astype(int)

# create the model
model = RandomForestClassifier()

# %%
X = data.drop('Stage' , axis=1)
y = data['Stage']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1) # 85% training and 15% test

# train the model
classifired_data = model.fit(X_train,y_train)

# write the model
f = open('RandomForest.pickle', 'wb+')
pickle.dump(classifired_data, f)
f.close()

# predict via the model
y_pred = classifired_data.predict(X_test)


# calculate accuracy and F1
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("F1:",metrics.f1_score(y_test, y_pred,average='macro'))

# %%
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

# %%

