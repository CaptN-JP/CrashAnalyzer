# Implement KNN along with the support and confidence and consider only the specified columns. All the columns are categorical.

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/crash_reporting_drivers_data_sanitized.csv')
# selected_cat_features = [
#     "Collision Type", "Weather", "Surface Condition", "Light", "Traffic Control", "Driver Substance Abuse",
#     "Non-Motorist Substance Abuse", "Driver At Fault",  "Circumstance", "Driver Distracted By",
#     "Drivers License State", "Vehicle Damage Extent", "Vehicle First Impact Location", "Vehicle Second Impact Location",
#     "Vehicle Body Type", "Vehicle Movement", "Vehicle Continuing Dir", "Vehicle Going Dir", "Speed Limit", "Driverless Vehicle",
#     "Parked Vehicle", "Vehicle Year", "Vehicle Make", "Vehicle Model", "Equipment Problems", "Latitude", "Longitude"
# ]

selected_cat_features = [
    "Collision Type", "Weather", "Surface Condition", "Light", "Traffic Control", "Driver Substance Abuse",
    "Non-Motorist Substance Abuse", "Driver At Fault", "Driver Distracted By", "Latitude", "Longitude"
]

data = df[selected_cat_features].copy()

# Cast all columns as string except "Latitude" and "Longitude"
for column in selected_cat_features:
    if column != "Latitude" and column != "Longitude":
        data[column] = data[column].astype(str)

data["Longitude"] = data["Longitude"].astype(str)
data["Latitude"] = data["Latitude"].astype(str)

# Split data into train and test sets
# X = data.drop('Injury Severity', axis=1)
X = data
y = df['Injury Severity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define predictors
predictors = X_train.columns.tolist()


# One-hot encoding
X = pd.get_dummies(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define predictors
predictors = X_train.columns.tolist()

# Create KNN classifier
knn = KNeighborsClassifier()
knn.fit(X_train[predictors], y_train)


# # Create KNN classifier
# knn = KNeighborsClassifier(n_neighbors=5)

# # Fit the classifier to the data
# knn.fit(X_train[predictors], y_train)

# Predict the labels for the test data
y_pred = knn.predict(X_test[predictors])

# Print the accuracy
print("Accuracy:", knn.score(X_test[predictors], y_test))

# Print the confusion matrix
print("Confusion Matrix:")
print(pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted']))

# Print the classification report
from sklearn.metrics import classification_report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print the top 10 features
print("Top 10 Features:")
print(knn.feature_importances_[:10])

# Print the bottom 10 features
print("Bottom 10 Features:")

