# class sklearn.tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
# max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort='deprecated', ccp_alpha=0.0)

# Parameters
# criterion{“gini”, “entropy”}, default=”gini”
# The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.

# splitter{“best”, “random”}, default=”best”
# The strategy used to choose the split at each node. Supported strategies are “best” to choose the best split and “random” to choose the best random split.

# max_depthint, default=None
# The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.

# min_samples_splitint or float, default=2
# The minimum number of samples required to split an internal node:

# If int, then consider min_samples_split as the minimum number.

# If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.

# Changed in version 0.18: Added float values for fractions.

# min_samples_leafint or float, default=1
# The minimum number of samples required to be at a leaf node. 
# A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. 
# This may have the effect of smoothing the model, especially in regression.

# If int, then consider min_samples_leaf as the minimum number.

# If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.



# Attributes
# classes_ndarray of shape (n_classes,) or list of ndarray
# The classes labels (single output problem), or a list of arrays of class labels (multi-output problem).

# feature_importances_ndarray of shape (n_features,)
# Return the feature importances.

# max_features_int
# The inferred value of max_features.

# n_classes_int or list of int
# The number of classes (for single output problems), or a list containing the number of classes for each output (for multi-output problems).

# n_features_int
# The number of features when fit is performed.

# n_outputs_int
# The number of outputs when fit is performed.

# tree_Tree
# The underlying Tree object. Please refer to help(sklearn.tree._tree.Tree) for attributes of Tree object and Understanding the decision tree structure for basic usage of these attributes.

from sklearn import tree
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

clf.predict([[2., 2.]])