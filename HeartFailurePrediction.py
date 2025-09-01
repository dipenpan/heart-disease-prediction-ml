#!/usr/bin/env python
# coding: utf-8

# ## LIBRARIES
# 

# In[496]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[497]:


from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix


# # Importing Dataset

# In[498]:


dataset = pd.read_csv('heart.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


# In[499]:


print("Dataset Shape:", dataset.shape)
dataset.head()


# # Data Preprocessing

# In[500]:


from sklearn.preprocessing import LabelEncoder
le1 = LabelEncoder() 
le2 = LabelEncoder() 
le6 = LabelEncoder() 
le8 = LabelEncoder() 
le10 = LabelEncoder() 
x[:,1] = le1.fit_transform(x[:,1])
x[:,2] = le2.fit_transform(x[:,2])
x[:,6] = le6.fit_transform(x[:,6])
x[:,8] = le8.fit_transform(x[:,8])
x[:,10] = le10.fit_transform(x[:,10])


# In[501]:


print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)



# ### Splitting Dataset into Training set and Test set

# In[502]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# # Feature Scaling

# In[503]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[504]:


print(X_test)


# # Training Dataset

# In[505]:


model_randomforest = RandomForestClassifier()
model_randomforest.fit(X_train, Y_train)



# In[506]:


model_logistic = LogisticRegression(max_iter=1000)
model_logistic.fit(X_train, Y_train)


# In[507]:


model_kneighbors = KNeighborsClassifier()
model_kneighbors.fit(X_train, Y_train)


# In[508]:


model_decision = DecisionTreeClassifier()
model_decision.fit(X_train, Y_train)


# In[509]:


model_svm = SVC()
model_svm.fit(X_train, Y_train)


# ###  Model Evaluation
# 

# In[510]:


from sklearn.metrics import confusion_matrix , accuracy_score
y_pred_logistic = model_logistic.predict(X_test)
y_pred_neighbors = model_kneighbors.predict(X_test)
y_pred_svm = model_svm.predict(X_test)
y_pred_decision = model_decision.predict(X_test)
y_pred_random = model_randomforest.predict(X_test)

RandomForest_Accuracy = accuracy_score(Y_test, model_randomforest.predict(X_test))
LogisticRegression_Accuracy = accuracy_score(Y_test, model_logistic.predict(X_test))
KNeighbors_Accuracy = accuracy_score(Y_test, model_kneighbors.predict(X_test))
DecisionTree_Accuracy = accuracy_score(Y_test, model_decision.predict(X_test))
SVM_Accuracy = accuracy_score(Y_test, model_svm.predict(X_test))

accuracy_dict = {
    "Logistic Regression": LogisticRegression_Accuracy,
    "KNeighbors": KNeighbors_Accuracy,
    "Support Vector Machine": SVM_Accuracy,
    "Random Forest": RandomForest_Accuracy,
    "Decision Tree": DecisionTree_Accuracy
}

# Plot accuracies
plt.figure(figsize=(12, 6))
plt.bar(accuracy_dict.keys(), accuracy_dict.values(), width=0.6, color='skyblue')
plt.xlabel("Machine Learning Algorithm")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison of ML Models")
plt.ylim(0, 1)
plt.show()

best_model_name = max(accuracy_dict, key=accuracy_dict.get)
models = {
    "Logistic Regression": model_logistic,
    "KNeighbors": model_kneighbors,
    "Support Vector Machine": model_svm,
    "Random Forest": model_randomforest,
    "Decision Tree": model_decision
}
best_model = models[best_model_name]

y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(Y_test, y_pred_best)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Failure", "Failure"],
            yticklabels=["No Failure", "Failure"])
plt.title(f"Confusion Matrix - {best_model_name}")
plt.show()

print(f"Best Model: {best_model_name} with Accuracy: {accuracy_dict[best_model_name]:.2f}")
print("Random Forest Accuracy:", RandomForest_Accuracy)


# In[511]:


print(RandomForest_Accuracy)


# ### Single Prediction

# In[512]:


def predict_single(patient_data, model=best_model):
    patient_array = np.array([patient_data], dtype=float)
    patient_array = scaler.transform(patient_array)
    prediction = model.predict(patient_array)[0]
    return "⚠️ Patient at Risk of Heart Failure" if prediction == 1 else "✅ Patient Healthy"



# Example Prediction

sample_patient = [65, 0, 582, 20, 0, 0, 130, 1, 0, 1.3, 111]  

print("Sample Prediction:", predict_single(sample_patient))


