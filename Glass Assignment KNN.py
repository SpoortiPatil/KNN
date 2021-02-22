o# Importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importing the dataset
glass= pd.read_csv("glass.csv")
glass.head()
glass.shape
glass.describe
glass.info()
glass.columns

glass.isna().sum()

#Univariate analysis
gls_features = glass.columns
gls_features

for feature in gls_features:
    if feature != "Type":
        glass[feature].hist(bins=25)
        plt.xlabel(feature)
        plt.ylabel("Counts")
        plt.title(feature)
        plt.show()
        
# Countplot for output variable
sns.countplot("Type", data=glass, palette="hls")

# Seperating input and output variables
X = glass.drop(columns="Type")
y = glass["Type"]

#Feature scaling
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X)

scaled_data = pd.DataFrame(scaler.transform(X), columns=X.columns )
scaled_data.head()

# Redefining X value
X = scaled_data

# Splitting the data into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
x_train.shape
x_test.shape

#Model building
from sklearn.neighbors import KNeighborsClassifier as KNC
#Assuming k = 3 initially
neigh = KNC(n_neighbors = 3)
model = neigh.fit(x_train, y_train)

# Predictions
pred_train = model.predict(x_train)        # for training
pred_test = model.predict(x_test)           # for testing

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Confusion matrix
confusion_matrix(y_train,pred_train)          # for training
confusion_matrix(y_test,pred_test)            # for testing

# model accuracy
acc_train = accuracy_score(y_train,pred_train)         # for training
acc_train
acc_test = accuracy_score(y_test,pred_test)         # for testing
acc_test

#Trying the value k =5
neigh = KNC(n_neighbors = 5)
model = neigh.fit(x_train, y_train)

# Predictions
pred_train5 = model.predict(x_train)        # for training
pred_test5 = model.predict(x_test)           # for testing

# Confusion matrix
confusion_matrix(y_train,pred_train5)          # for training
confusion_matrix(y_test,pred_test5)           # for testing

# model accuracy
acc_train = accuracy_score(y_train,pred_train5)         # for training
acc_train

acc_test = accuracy_score(y_test,pred_test5)         # for testing
acc_test

#Checking the accuracy of the model for different value of k
acc=[]
for i in range(3,20,2):
    neigh=KNC(n_neighbors=i)
    model = neigh.fit(x_train, y_train)
    pred_trainf=model.predict(x_train)
    pred_testf=model.predict(x_test)
    train_acc = accuracy_score(y_train,pred_trainf)
    test_acc=accuracy_score(y_test,pred_testf)
    acc.append([train_acc, test_acc])
 
# Plotting accuracy of the model considering k values from 3 to 20
plt.plot(np.arange(3,20,2), [i[0] for i in acc],"ro-")    

plt.plot(np.arange(3,20,2), [i[1] for i in acc], "bo-")

# Accuracy of the KNN model for k =3
acc_train = accuracy_score(y_train,pred_trainf)         # for training
acc_train

acc_test = accuracy_score(y_test,pred_testf)         # for testing
acc_test
