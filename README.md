# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.The required Python libraries are imported and the dataset is loaded, after which the input features and target variable are separated.

2.Categorical data is converted into numerical form using one-hot encoding and the dataset is divided into training and testing sets.

3.The input features are scaled using Standard Scaler and an SGD Classifier model is created.

4.The model is trained using the training data, predictions are made on test data, and the accuracy and performance metrics are calculated.

5.Finally, a new data sample is scaled using the same scaler and its class is predicted using the trained model.
## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
# 1️⃣ Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2️⃣ Load Dataset
data = pd.read_csv("Placement_Data.csv")

# 3️⃣ Separate Features and Target
X = data.drop(["status", "salary", "sl_no"], axis=1)
y = data["status"]

# 4️⃣ One-Hot Encoding for Categorical Columns
X = pd.get_dummies(X, drop_first=True)

# 5️⃣ Save Feature Names
feature_names = X.columns

# 6️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7️⃣ Feature Scaling (keep as DataFrame)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=feature_names
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=feature_names
)

# 8️⃣ Create SGD Classifier
model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)

# 9️⃣ Train Model
model.fit(X_train_scaled, y_train)

# 🔟 Predict on Test Data
y_pred = model.predict(X_test_scaled)

# 1️⃣1️⃣ Accuracy & Reports
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ============================
# 1️⃣2️⃣ Correct New Student Prediction
# ============================

# Create empty student with all features = 0
new_student_dict = dict.fromkeys(feature_names, 0)

# Fill ONLY numerical features (example values)
new_student_dict['ssc_p'] = 67
new_student_dict['hsc_p'] = 91
new_student_dict['degree_p'] = 58
new_student_dict['etest_p'] = 88
new_student_dict['mba_p'] = 67

# Convert to DataFrame with same columns
new_student_df = pd.DataFrame([new_student_dict])

# Scale using same scaler
new_student_scaled = pd.DataFrame(
    scaler.transform(new_student_df),
    columns=feature_names
)

# Predict
pred = model.predict(new_student_scaled)
print("\nPredicted Status:", pred[0])
Developed by: DIVYA K
RegisterNumber: 25019198
*/
```

## Output:
![prediction of iris species using SGD Classifier]
<img width="636" height="411" alt="image" src="https://github.com/user-attachments/assets/f1fd4e6c-41e7-4d50-bd64-45cf13d531dd" />



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
