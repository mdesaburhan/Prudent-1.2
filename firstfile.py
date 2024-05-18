# Load the dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from imblearn.combine import SMOTEENN
import joblib

# Load the dataset using the absolute file path
ch = pd.read_csv('C:/Users/MOHAMMED ESA/Downloads/churnprudent.csv')

# Assuming 'Churn' is the target column
X = ch.drop(columns=['Churn'])
y = ch['Churn']

# Perform one-hot encoding on categorical variables
X = pd.get_dummies(X)

# Resample the dataset using SMOTEENN
smoteenn = SMOTEENN()
X_resampled, y_resampled = smoteenn.fit_resample(X, y)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train your model (example using Decision Tree Classifier)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Print feature importances
print("Feature Importances:")
for feature, importance in zip(X_train.columns, model.feature_importances_):
    print(f"{feature}: {importance}")

# Evaluate model performance (example using accuracy)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save your trained model to disk
joblib.dump(model, 'decision_tree_model.pkl')
