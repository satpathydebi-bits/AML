from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from decission_tree_classifier import X_train
from decission_tree_classifier import y_train
from decission_tree_classifier import X_val
from decission_tree_classifier import y_val


# Implement Random Forest Classifier
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)

# Evaluate the model on the validation set
y_pred = rf_clf.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy on the validation set: {accuracy:.2f}")

# Compare the results with Decision Tree Classifier
print(f"Decision Tree Classifier Accuracy: {accuracy:.2f}")
print(f"Random Forest Classifier Accuracy: {accuracy:.2f}")