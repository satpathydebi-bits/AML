import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset from CSV
iris_df = pd.read_csv('C:\\Users\\91700\\Dropbox\\My PC (LAPTOP-J312LENU)\\Desktop\\BITS\\Sem-2\\AML-2\\Iris.csv')
# Separate features and target variable
X = iris_df.drop('Species', axis=1)
y = iris_df['Species']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Check if accuracy is more than 90%
if accuracy > 0.90:
    print("Performance score achieved!")
else:
    print("Performance score not achieved.")
