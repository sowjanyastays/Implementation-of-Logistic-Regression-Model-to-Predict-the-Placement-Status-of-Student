# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import dataset
2. Check for null and duplicate values
3. Assign x and y values
4. Split data into train and test data
5. Import logistic regression and fit the training data
6. Predict y value
7. Calculate accuracy and confusion matrix

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Sowjanya S
RegisterNumber:  212220040158
*/
import pandas as pd

data = pd.read_csv("Placement_Dataa.csv")

print("1. Placement data")
data.head()

data1 = data.copy()
data1= data1.drop(["sl_no","salary"],axis=1) #feature selection
print("2. Salary Data")
print(data1.head())

print("3. Checking the null() function")
data1.isnull().sum()

print("4. Data Duplicate")
data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
lc = LabelEncoder() #changing values
data1["gender"] = lc.fit_transform(data1["gender"])
data1["ssc_b"] = lc.fit_transform(data1["ssc_b"])
data1["hsc_b"] = lc.fit_transform(data1["hsc_b"])
data1["hsc_s"] = lc.fit_transform(data1["hsc_s"])
data1["degree_t"]=lc.fit_transform(data["degree_t"])
data1["workex"] = lc.fit_transform(data1["workex"])
data1["specialisation"] = lc.fit_transform(data1["specialisation"])
data1["status"]=lc.fit_transform(data1["status"])

print("5. Print data")
print(data1)

y = data1["status"]
print("6. Data-status")
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear")
print("7. y_prediction array")
print(lr.fit(x_train,y_train))

y_pred = lr.predict(x_test)
print(y_pred)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print("8. Accuracy")
print(accuracy)

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
print("9. Confusion array")
print(confusion)

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print("10. Classification report")
print(classification_report1)
prediction = [1,80,1,90,1,1,90,1,0,85,1,85]

print("11. Prediction of LR")
print(lr.predict([prediction]))

```

## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](output1.png)
![the Logistic Regression Model to Predict the Placement Status of Student](output2.png)
![the Logistic Regression Model to Predict the Placement Status of Student](output3.png)
![the Logistic Regression Model to Predict the Placement Status of Student](output4.png)
![the Logistic Regression Model to Predict the Placement Status of Student](output5.png)
![the Logistic Regression Model to Predict the Placement Status of Student](output6.png)
![the Logistic Regression Model to Predict the Placement Status of Student](output7.png)
![the Logistic Regression Model to Predict the Placement Status of Student](output8.png)
![the Logistic Regression Model to Predict the Placement Status of Student](output9.png)
![the Logistic Regression Model to Predict the Placement Status of Student](output10.png)
![the Logistic Regression Model to Predict the Placement Status of Student](output11.png)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
