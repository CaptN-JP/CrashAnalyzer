# Implement Random Forest Classification Algorithm to predict the severity of the crash.

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from spatial_viz import spatial_viz
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from imblearn.over_sampling import SMOTE
# Read the data
df = pd.read_csv('data/crash_reporting_drivers_data_sanitized.csv')
selected_cat_features = [
    "Collision Type", "Weather", "Surface Condition", "Light", "Traffic Control", "Driver Substance Abuse",
    "Non-Motorist Substance Abuse", "Driver At Fault",  "Circumstance", "Driver Distracted By",
    "Drivers License State", "Vehicle Damage Extent", "Vehicle First Impact Location", "Vehicle Second Impact Location",
    "Vehicle Body Type", "Vehicle Movement", "Vehicle Continuing Dir", "Vehicle Going Dir", "Speed Limit", "Driverless Vehicle",
    "Parked Vehicle", "Vehicle Year", "Vehicle Make", "Vehicle Model", "Equipment Problems", "Latitude", "Longitude"
]

data = df[selected_cat_features].copy()

# Convert categorical features to numerical features
data = pd.get_dummies(data)

# Split the data into training and testing sets
X = data
y = df['Injury Severity']

# Apply SMOTE to address class imbalance
smote = SMOTE(random_state=0)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, random_state=0)

# Create a random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=0)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{cm}")

# Visualize the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

