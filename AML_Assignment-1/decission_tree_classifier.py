import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the Titanic dataset
test_data = pd.read_csv('D:\\codes\\AML_Assignment-1\\titanic_test.csv')
train_data = pd.read_csv('D:\\codes\\AML_Assignment-1\\titanic_train.csv')

# Separate features and target variable
X_train = train_data.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
y_train = train_data['Survived']
test_data = test_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Fill in missing values for numeric columns
numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
X_train[numeric_cols] = X_train[numeric_cols].fillna(X_train[numeric_cols].mean())
test_data[numeric_cols] = test_data[numeric_cols].fillna(test_data[numeric_cols].mean())

# Encode categorical features
le = LabelEncoder()
X_train['Sex'] = le.fit_transform(X_train['Sex'])
test_data['Sex'] = le.transform(test_data['Sex'])
X_train['Embarked'] = le.fit_transform(X_train['Embarked'])
test_data['Embarked'] = le.transform(test_data['Embarked'])

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train the Decision Tree Classifier
dt_clf = DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42)
dt_clf.fit(X_train, y_train)

# Evaluate the model
y_pred = dt_clf.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f'Decision Tree Classifier Accuracy: {accuracy:.2f}')
