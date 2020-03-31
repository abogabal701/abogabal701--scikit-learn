# class sklearn.svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, 
# probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, 
# decision_function_shape='ovr', break_ties=False, random_state=None)

# Parameters:

    # Cfloat, optional (default=1.0)
    # Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty.

    # kernelstring, optional (default=’rbf’)
    # Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. If none is given, 
    # ‘rbf’ will be used. If a callable is given it is used to pre-compute the kernel matrix from data matrices; that matrix should be an array of shape (n_samples, n_samples).

    # degreeint, optional (default=3)
    # Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.

    # gamma{‘scale’, ‘auto’} or float, optional (default=’scale’)
    # Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.

    # if gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma,

    # if ‘auto’, uses 1 / n_features.

    # Changed in version 0.22: The default value of gamma changed from ‘auto’ to ‘scale’.

    # coef0float, optional (default=0.0)
    # Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.

    # shrinkingboolean, optional (default=True)
    # Whether to use the shrinking heuristic.

    # probabilityboolean, optional (default=False)
    # Whether to enable probability estimates. This must be enabled prior to calling fit, 
    # will slow down that method as it internally uses 5-fold cross-validation, and predict_proba may be inconsistent with predict. Read more in the User Guide.

    # tolfloat, optional (default=1e-3)
    # Tolerance for stopping criterion.

    # cache_sizefloat, optional
    # Specify the size of the kernel cache (in MB).

    # class_weight{dict, ‘balanced’}, optional
    # Set the parameter C of class i to class_weight[i]*C for SVC. If not given, all classes are supposed to have weight one. 
    # The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))


    # Attributes
    #         support_array-like of shape (n_SV)
    #         Indices of support vectors.

    #         support_vectors_array-like of shape (n_SV, n_features)
    #         Support vectors.

    #         n_support_array-like, dtype=int32, shape = [n_class]
    #         Number of support vectors for each class.

    #         dual_coef_array, shape = [n_class-1, n_SV]
    #         Coefficients of the support vector in the decision function. For multiclass, 
    # coefficient for all 1-vs-1 classifiers. The layout of the coefficients in the multiclass case is somewhat non-trivial. 
    # See the section about multi-class classification in the SVM section of the User Guide for details.

    #         coef_array, shape = [n_class * (n_class-1) / 2, n_features]
    #         Weights assigned to the features (coefficients in the primal problem). This is only available in the case of a linear kernel.

    #         coef_ is a readonly property derived from dual_coef_ and support_vectors_.

    #         intercept_ndarray of shape (n_class * (n_class-1) / 2,)
    #         Constants in decision function.

    #         fit_status_int
    #         0 if correctly fitted, 1 otherwise (will raise warning)

    #         classes_array of shape (n_classes,)
    #         The classes labels.


import numpy as np
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])
from sklearn.svm import SVC
clf = SVC(gamma='auto')
clf.fit(X, y)
SVC(gamma='auto')
print(clf.predict([[-0.8, -1]]))