import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# 1. Load the dataset
data = pd.read_csv("C:\\Users\\Vasantha Kamalee\\Downloads\\tested.csv")

# 2. Quick look
print(data.head())
print(data.info())

# 3. Drop unnecessary columns (if any)
# Example: PassengerId, Name, Ticket are usually dropped
data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True, errors='ignore')

# 4. Handle missing values
# Fill missing 'Age' with median
# Fill missing 'Age' with median
age_imputer = SimpleImputer(strategy='median')
data['Age'] = age_imputer.fit_transform(data[['Age']]).ravel()

# Fill missing 'Fare' with median (you have 1 missing value in Fare too!)
fare_imputer = SimpleImputer(strategy='median')
data['Fare'] = fare_imputer.fit_transform(data[['Fare']]).ravel()

# Fill missing 'Embarked' with most frequent
embarked_imputer = SimpleImputer(strategy='most_frequent')
data['Embarked'] = embarked_imputer.fit_transform(data[['Embarked']]).ravel()

# 5. Encode categorical variables
# 'Sex' and 'Embarked' are categorical
label_encoders = {}
for column in ['Sex', 'Embarked']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# 6. Split features and target
X = data.drop('Survived', axis=1)
y = data['Survived']

# 7. Normalize numerical features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 8. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 9. Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 10. Make predictions
y_pred = model.predict(X_test)

# 11. Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
