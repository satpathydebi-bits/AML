import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Load the dataset
data = pd.read_csv("C:\\Users\\91700\\Dropbox\\My PC (LAPTOP-J312LENU)\\Desktop\\BITS\\Sem-2\\AML-2\\data_breast_cancer.csv")

# Preprocess the data
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})  # Encode labels
X = data.iloc[:, 2:]  # Features
y = data['diagnosis']  # Labels

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Evaluate the model
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
