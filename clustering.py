# Implement clustering algorithm: Catboost

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from catboost import Pool
from imblearn.over_sampling import SMOTE

df = pd.read_csv('data/crash_reporting_drivers_data_sanitized.csv')
selected_cat_features = [
    "Collision Type", "Weather", "Surface Condition", "Light", "Traffic Control", "Driver Substance Abuse",
    "Non-Motorist Substance Abuse", "Driver At Fault",  "Circumstance", "Driver Distracted By",
    "Drivers License State", "Vehicle Damage Extent", "Vehicle First Impact Location", "Vehicle Second Impact Location",
    "Vehicle Body Type", "Vehicle Movement", "Vehicle Continuing Dir", "Vehicle Going Dir", "Speed Limit", "Driverless Vehicle",
    "Parked Vehicle", "Vehicle Year", "Vehicle Make", "Vehicle Model", "Equipment Problems", "Latitude", "Longitude"
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
# smote = SMOTE(random_state=0)
# X_resampled, y_resampled = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define predictors
predictors = X_train.columns.tolist()

# Prepare model
clf = CatBoostClassifier(iterations=600, eval_metric='Accuracy', verbose=50)
# Fit model
clf.fit(X_train[predictors], y_train, eval_set=(X_test, y_test), cat_features=selected_cat_features)

# Get predictions
preds_class = clf.predict(X_test)
preds_proba = clf.predict_proba(X_test)

# Get predicted classes
preds_class = clf.predict(X_test)

# Get predicted probabilities for each class
preds_proba = clf.predict_proba(X_test)

# Get predicted RawFormulaVal
preds_raw = clf.predict(X_test, prediction_type='RawFormulaVal')

# Print the results
print("class = ", preds_class)
print("proba = ", preds_proba)
print("RawFormulaVal = ", preds_raw)

# Get the accuracy score
print("Accuracy = ", clf.score(X_test, y_test))

# Get the confusion matrix
cm = confusion_matrix(y_test, preds_class)
print("Confusion Matrix:\n", cm)

# Get the feature importance
# feature_importances = clf.get_feature_importance(X=X_train[predictors], y=y_train, cat_features=selected_cat_features)
# feature_names = X_train[predictors].columns
# Create a Pool object
train_pool = Pool(data=X_train[predictors], label=y_train, cat_features=selected_cat_features)
feature_importances = clf.get_feature_importance(train_pool)
for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
    print('{}: {}'.format(name, score))


# Get the feature importance graph
sns.set()
plt.figure(figsize=(10, 7))
sns.barplot(x=feature_importances, y=feature_names)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

# Get the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, preds_proba[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# Get the precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, preds_proba[:, 1])
plt.figure(figsize=(10, 7))
plt.plot(recall, precision, color='darkorange', lw=2, label='Precision-Recall curve')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower right')
plt.show()

# Save the model
clf.save_model('catboost_model.dump')
