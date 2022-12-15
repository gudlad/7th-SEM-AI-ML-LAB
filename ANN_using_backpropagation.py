import numpy as np
# Features(Hrs Slept,Hrs Studied)
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)  # Labels(Marks Obtained)
X = X/np.amax(X, axis=0)  # Normalize
y = y/100


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_grad(x):
    return x*(1-x)


# variable initialization
epoch = 1000  # Setting training iterations
eta = 0.2     # Setting learning rate(eta)
input_neurons = 2   # number of features in data set
hidden_neurons = 3  # number of hidden layers neurons
output_neurons = 1  # number of neurons at output layer
# weight and bias random initialization
wh = np.random.uniform(size=(input_neurons, hidden_neurons))
bh = np.random.uniform(size=(1, hidden_neurons))
wout = np.random.uniform(size=(hidden_neurons, output_neurons))
bout = np.random.uniform(size=(1, output_neurons))
for i in range(epoch):
    # forward propogation
    h_ip = np.dot(X, wh)+bh
    h_act = sigmoid(h_ip)
    o_ip = np.dot(h_act, wout)+bout
    output = sigmoid(o_ip)
# Back propogation
# Error at output layer
    Eo = y-output
    outgrad = sigmoid_grad(output)
    d_output = Eo*outgrad
# Error at hidden layer
    Eh = d_output.dot(wout.T)
    hiddengrad = sigmoid_grad(h_act)
    d_hidden = Eh*hiddengrad
    wout += h_act.T.dot(d_output)*eta
    wh += X.T.dot(d_hidden)*eta
print("Normalized Input:\n"+str(X))
print("Actual Output:\n"+str(y))
print("Predicted output:\n", output)
