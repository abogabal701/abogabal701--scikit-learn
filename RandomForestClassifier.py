#class sklearn.ensemble.RandomForestClassifier(
#    n_estimators=100, 
#    criterion='gini', 
#    max_depth=None,
#    min_samples_split=2, 
#    min_samples_leaf=1, 
#    min_weight_fraction_leaf=0.0, 
#    max_features='auto', 
#    max_leaf_nodes=None, 
#    min_impurity_decrease=0.0, 
#    min_impurity_split=None, 
#    bootstrap=True, 
#    oob_score=False, 
#    n_jobs=None, 
#    random_state=None, 
#    verbose=0, 
#    warm_start=False, 
#    class_weight=None, 
#    ccp_alpha=0.0, 
#    max_samples=None)


#   A random forest classifier.

#   A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.

#   The sub-sample size is always the same as the original input sample size but the samples are drawn with replacement if bootstrap=True (default).


#       Parameters:

                    ####    n_estimators integer, optional (default=100)
                    #           The number of trees in the forest.
                    #           Changed in version 0.22: The default value of n_estimators changed from 10 to 100 in 0.22.

                    ####    criterionstring, optional (default=”gini”)
                    #           The function to measure the quality of a split. 
                    #           Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain. Note: this parameter is tree-specific.

                    ####    max_depthinteger or None, optional (default=None)
                    #           The maximum depth of the tree. 
                    #           If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.




                    ####     warm_startbool, optional (default=False)
                    #           When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new forest. See the Glossary.

                    ####    class_weightdict, list of dicts, “balanced”, “balanced_subsample” or None, optional (default=None)
                    #           Weights associated with classes in the form {class_label: weight}. If not given, 
                    #           all classes are supposed to have weight one. For multi-output problems, a list of dicts can be provided in the same order as the columns of y.
                    #           Note that for multioutput (including multilabel) weights should be defined for each class of every column in its own dict. For example, 
                    #           for four-class multilabel classification weights should be [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of [{1:1}, {2:5}, {3:1}, {4:1}].
                    #           The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))
                    #           The “balanced_subsample” mode is the same as “balanced” except that weights are computed based on the bootstrap sample for every tree grown.
                    #           For multi-output, the weights of each column of y will be multiplied.
                    #           Note that these weights will be multiplied with sample_weight (passed through the fit method) if sample_weight is specified.

                    ####    ccp_alphanon-negative float, optional (default=0.0)
                    #           Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is smaller than ccp_alpha will be chosen. 
                    #           By default, no pruning is performed. See Minimal Cost-Complexity Pruning for details.




#       Attributes:

                    ####    base_estimator_DecisionTreeClassifier
                    #           The child estimator template used to create the collection of fitted sub-estimators.

                    #### estimators_list of DecisionTreeClassifier
                    #           The collection of fitted sub-estimators.

                    #### classes_array of shape (n_classes,) or a list of such arrays
                    #           The classes labels (single output problem), or a list of arrays of class labels (multi-output problem).

                    #### n_classes_int or list
                    #           The number of classes (single output problem), or a list containing the number of classes for each output (multi-output problem).

                    #### n_features_int
                    #           The number of features when fit is performed.

                    #### n_outputs_int
                    #           The number of outputs when fit is performed.

                    ####feature_importances_ndarray of shape (n_features,)
                    #           Return the feature importances (the higher, the more important the feature).

                    #### oob_score_float
                    #           Score of the training dataset obtained using an out-of-bag estimate. This attribute exists only when oob_score is True.

                    #### oob_decision_function_array of shape (n_samples, n_classes)
                    #           Decision function computed with out-of-bag estimate on the training set. 
                    #           If n_estimators is small it might be possible that a data point was never left out during the bootstrap. 
                    #           In this case, oob_decision_function_ might contain NaN. This attribute exists only when oob_score is True.


#Examples:

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


X, y = make_classification(n_samples=1000, n_features=4,
                           n_informative=2, n_redundant=0,
                          random_state=0, shuffle=False)
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, y)
RandomForestClassifier(max_depth=2, random_state=0)
print(clf.feature_importances_)
print(clf.predict([[0, 0, 0, 0]]))
