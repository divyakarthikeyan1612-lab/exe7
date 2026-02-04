# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the Iris dataset.

2.Convert the data into features and target labels.

3.Split the dataset into training and testing sets.

4.Train an SGD classifier using the training data.

5.Predict the test data and evaluate the model using accuracy and confusion matrix.
## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
/*
/*
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
iris_data = load_iris()
iris_df = pd.DataFrame(
    iris_data.data,
    columns=iris_data.feature_names
)
iris_df["label"] = iris_data.target
print(iris_df.sample(5))
features = iris_df.iloc[:, :-1]
labels = iris_df.iloc[:, -1]
X_tr, X_te, y_tr, y_te = train_test_split(
    features,
    labels,
    test_size=0.2,
    random_state=1
)
model = SGDClassifier(
    max_iter=1200,
    tol=0.001
)
model.fit(X_tr, y_tr)
predictions = model.predict(X_te)
acc = accuracy_score(y_te, predictions)
print("Model Accuracy:", round(acc, 3))
conf_mat = confusion_matrix(y_te, predictions)
print("Confusion Matrix:")
print(conf_mat)

Developed by: DIVYA K
RegisterNumber:  25019198
*/
```

## Output:
![prediction of iris species using SGD Classifier]
<img width="856" height="401" alt="image" src="https://github.com/user-attachments/assets/1cc12974-56e7-4c9f-b3aa-94d9b0aa2ad3" />




## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
