#class sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', 
# solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', 
# learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, 
# tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, 
# early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)
# Parameters:
#   hidden_layer_sizestuple, length = n_layers - 2, default=(100,)
#       The ith element represents the number of neurons in the ith hidden layer.

#   activation{‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default=’relu’
#       Activation function for the hidden layer.

#   ‘identity’, no-op activation, useful to implement linear bottleneck, returns f(x) = x

#       ‘logistic’, the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).

#       ‘tanh’, the hyperbolic tan function, returns f(x) = tanh(x).

#       ‘relu’, the rectified linear unit function, returns f(x) = max(0, x)

#       solver{‘lbfgs’, ‘sgd’, ‘adam’}, default=’adam’
#       The solver for weight optimization.

#       ‘lbfgs’ is an optimizer in the family of quasi-Newton methods.

#       ‘sgd’ refers to stochastic gradient descent.

#       ‘adam’ refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba

#       Note: The default solver ‘adam’ works pretty well on relatively large datasets (with thousands of training samples or more) in terms of both training time and validation score. For small datasets, 
#       however, ‘lbfgs’ can converge faster and perform better.

#       alphafloat, default=0.0001
#       L2 penalty (regularization term) parameter.

#       batch_sizeint, default=’auto’
#       Size of minibatches for stochastic optimizers. If the solver is ‘lbfgs’, the classifier will not use minibatch. When set to “auto”, batch_size=min(200, n_samples)

#       learning_rate{‘constant’, ‘invscaling’, ‘adaptive’}, default=’constant’
#       Learning rate schedule for weight updates.

#       ‘constant’ is a constant learning rate given by ‘learning_rate_init’.

#       ‘invscaling’ gradually decreases the learning rate at each time step ‘t’ using an inverse scaling exponent of ‘power_t’. effective_learning_rate = learning_rate_init / pow(t, power_t)

#       ‘adaptive’ keeps the learning rate constant to ‘learning_rate_init’ as long as training loss keeps decreasing. Each time two consecutive epochs fail to decrease training loss by at least tol, or fail to increase validation score by at least tol if ‘early_stopping’ is on, the current learning rate is divided by 5.

#       Only used when solver='sgd'.

# Attributes:

# classes_ndarray or list of ndarray of shape (n_classes,)
#   Class labels for each output.

# loss_float
#   The current loss computed with the loss function.

# coefs_list, length n_layers - 1
#   The ith element in the list represents the weight matrix corresponding to layer i.

# intercepts_list, length n_layers - 1
#   The ith element in the list represents the bias vector corresponding to layer i + 1.

# n_iter_int,
#   The number of iterations the solver has ran.

# n_layers_int
#   Number of layers.

# n_outputs_int
#   Number of outputs.

# out_activation_string
#   Name of the output activation function.






from sklearn.neural_network import MLPClassifier
X = [[0., 0.], [1., 1.]]
y = [0, 1]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                   hidden_layer_sizes=(5, 2), random_state=1)

clf.fit(X, y)
MLPClassifier(alpha=1e-05, hidden_layer_sizes=(5, 2), random_state=1,
              solver='lbfgs')



clf.predict([[2., 2.], [-1., -2.]])




