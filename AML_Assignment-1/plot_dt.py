import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from decission_tree_classifier import dt_clf
from decission_tree_classifier import X_train

# Visualize the Decision Tree
plt.figure(figsize=(12, 12))
plot_tree(dt_clf, feature_names=X_train.columns, class_names=['Died', 'Survived'], filled=True)
plt.savefig('D:\\codes\\AML_Assignment-1\\decision_tree.png')