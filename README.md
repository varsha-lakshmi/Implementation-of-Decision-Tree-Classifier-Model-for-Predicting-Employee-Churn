# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: VARSHA S
RegisterNumber: 212225040482
*/
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
df = pd.read_csv("C:/Users/acer/Downloads/Employee.csv")
df.head()
df.info()
df.isnull().sum()
df.rename(columns={'Departments ': 'Department'}, inplace=True)
df = pd.get_dummies(df, columns=['Department', 'salary'], drop_first=True)
df.head()
X = df.drop('left', axis=1)  
y = df['left']  
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
dt_model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=5,
    random_state=42
)
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))
plot_tree(
    dt_model,
    feature_names=X.columns,
    class_names=['Stayed', 'Left'],
    filled=True
)
plt.show()
```

## Output:


<img width="944" height="556" alt="Screenshot 2026-08-17 213602" src="https://github.com/user-attachments/assets/24b6e0b4-1e56-4c7e-a40b-42f0459dfbdd" />



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
