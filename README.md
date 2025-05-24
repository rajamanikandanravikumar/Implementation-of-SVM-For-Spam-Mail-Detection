# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages.
2. Analyse the data.
3. Use modelselection and Countvectorizer to preditct the values.
4. Find the accuracy and display the result.
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Rajamanikandan R
RegisterNumber:  212223220082
*/
```
```
import pandas as pd
data=pd.read_csv("spam.csv", encoding='Windows-1252')
data

data.shape

x=data['v2'].values
y=data['v1'].values
x.shape

print(y.shape)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

print(x_train.shape)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
print(acc)

con=confusion_matrix(y_test,y_pred)
print(con)

cl=classification_report(y_test,y_pred)
print(cl)
```

## Output:

### x.shape() and y.shape()
![image](https://github.com/user-attachments/assets/f0c124b4-2a41-4e78-8f6b-4d8b6e636065)

### acc (accuracy)
![image](https://github.com/user-attachments/assets/eca8c2c7-71a1-44bd-88bc-e690fd9cea8e)

### con (confusion matrix)
![image](https://github.com/user-attachments/assets/d3e8edfe-d0d7-4efd-addf-50774a4effb8)

### cl (classification report)
![image](https://github.com/user-attachments/assets/10e75ed1-9dfb-4f39-942a-26a38c98b3c5)




## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
