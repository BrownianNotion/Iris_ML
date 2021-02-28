import pandas as pd
import numpy as np 
from scipy.optimize import minimize

#Read and shuffle data
iris_df = pd.read_csv("iris.csv")
species_names = iris_df['Species'].unique()
iris_shuffled_df = iris_df.sample(frac = 1)

#SECTION 1: Split into training and test examples
m_train = 120
m_test = len(iris_df.index) - m_train
iris_train = iris_shuffled_df.head(m_train)
iris_test = iris_shuffled_df.tail(m_test)

#Convert to X 
def df_to_X(df, drops = ['Id', 'Species']):
    X = df.drop(drops, 1).to_numpy()
    m = X.shape[0]
    X = np.c_[np.ones(m), X]
    return X

#Convert data frame to y vector/matrix of 1's and 0s
#each row is a training example
def df_to_y(df, species_names, drops = ['Id', 'Species']):
    m = len(df.index)
    k = len(species_names)
    y = np.zeros((m, k))
    for i in range(k):
        y_temp = df['Species'] == species_names[i] 
        y_temp = y_temp.to_numpy()
        y[:, i] = y_temp
    return y

#SECTION 2: Randomly Initialise weights
input_layer_size = 4 #does not include bias unit
hidden_layer_size = 4 #does not include bias unit
output_layer_size = len(species_names)

def rand_initialise_weights(L_in, L_out):
    epsilon = 0.12 #small number to base random initialisation on
    return np.random.rand(L_out, L_in + 1) * 2 * epsilon - epsilon

#Initialise random weights for Theta1, Theta2, then unroll into a big vector
Theta1_initial = rand_initialise_weights(input_layer_size, hidden_layer_size)
Theta2_initial = rand_initialise_weights(hidden_layer_size, output_layer_size)

Theta_initial_unrolled = np.concatenate((np.ravel(Theta1_initial), np.ravel(Theta2_initial)))

#SECTION 3: Forward and Backward Propagation functions
def sigmoid(z):
    return 1/(1 + np.exp(-z)) 

#Sigmoid gradient; the derivative of sigmoid function
def Dsigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))

#Function that calculates either cost function or gradients depending on second argument
def cost_gradient(Theta, return_val, X, Y,
         input_layer_size, hidden_layer_size, output_layer_size,
         lamb = 0):
    #reform Theta1/2 matrices
    Theta1_vec = Theta[0:((input_layer_size + 1) * hidden_layer_size)]
    Theta2_vec = Theta[((input_layer_size + 1) * hidden_layer_size):]
    Theta1 = np.reshape(Theta1_vec, (hidden_layer_size, input_layer_size + 1))
    Theta2 = np.reshape(Theta2_vec, (output_layer_size, hidden_layer_size + 1))

    ##########################################
    #FORWARD PROPAAGATION TO GET HYPOTHESIS
    ##########################################
    #no. of training examples
    m = X.shape[0]
    #1st Layer
    a1 = X
    #Layer 2
    z2 = a1.dot(Theta1.T)
    a2 = sigmoid(z2)
    a2 = np.c_[np.ones(m), a2]
    #Layer 3
    z3 = a2.dot(Theta2.T)
    H = sigmoid(z3) # H = hypothesis matrix

    if return_val == 'Cost':
        cost1 = np.trace( Y.T.dot(np.log(H)) )
        cost2 = np.trace( (1 - Y).T.dot(np.log(1 - H)) )
        J = -1/m * (cost1.item() + cost2.item()) #item converts from 1 by 1 numpy array to scalar
        if lamb == 0:
            return J
        
        #Remove first columns of Theta1, Theta2
        Theta1 = Theta1[:, 1:]
        Theta2 = Theta2[:, 1:]
        #regularisation term
        reg_term = lamb/(2 * m) * (np.sum(Theta1 * Theta1) + np.sum(Theta2 * Theta2))  
        return J + reg_term
    elif return_val == 'Gradient':
        delta3 = H - Y
        delta2 = delta3.dot(Theta2[:, 1:]) * Dsigmoid(z2)
        Delta1 = delta2.T.dot(a1)
        Delta2 = delta3.T.dot(a2)

        #Calculate gradients
        Theta1_grad = 1/m * Delta1
        Theta2_grad = 1/m * Delta2

        #Add regularisation terms. Remember that first column is not reguarlised 
        Theta1_grad[:, 1:] = Theta1_grad[:, 1:] + lamb/m * Theta1[:, 1:]
        Theta2_grad[:, 1:] = Theta2_grad[:, 1:] + lamb/m * Theta2[:, 1:]

        #unroll into a single vector
        Theta_grad = np.concatenate((np.ravel(Theta1_grad), np.ravel(Theta2_grad)))
        return Theta_grad
    else:
        print("Invalid return type")
        return

def cost(Theta, X, Y,
              input_layer_size, hidden_layer_size, output_layer_size,
              lamb = 0):
    return cost_gradient(Theta, 'Cost', X, Y,
              input_layer_size, hidden_layer_size, output_layer_size,
              lamb)

def gradients(Theta, X, Y,
              input_layer_size, hidden_layer_size, output_layer_size,
              lamb = 0):
    return cost_gradient(Theta, 'Gradient', X, Y,
              input_layer_size, hidden_layer_size, output_layer_size,
              lamb)


#Process Training data and train the network
X_train = df_to_X(iris_train)
Y_train = df_to_y(iris_train, species_names)

#Regularisation parameter
lamb = 0.01

#Minimise the cost function using BFGS method, providing a gradient function
res = minimize(cost, Theta_initial_unrolled, args = (X_train, Y_train, input_layer_size, hidden_layer_size, output_layer_size, lamb),
               method = 'BFGS', jac = gradients, options = {'disp': True})

#Reshape back into theta matrices
boundary = (input_layer_size + 1) * hidden_layer_size
Theta1_trained = res.x[0:boundary]
Theta1_trained = np.reshape(Theta1_trained, (hidden_layer_size, input_layer_size + 1))
Theta2_trained = res.x[boundary:]
Theta2_trained = np.reshape(Theta2_trained, (output_layer_size, hidden_layer_size + 1))


#Calculates hypothesis
def hypothesis(X, Theta1, Theta2):
    #no. of training examples
    m = X.shape[0]
    #1st Layer
    a1 = X
    #Layer 2
    z2 = a1.dot(Theta1.T)
    a2 = sigmoid(z2)
    a2 = np.c_[np.ones(m), a2]
    #Layer 3
    z3 = a2.dot(Theta2.T)
    return sigmoid(z3)

#Function returns prediction vector with labels 0,1,2 and prints the accuracy of the predictions
#Y is a mx3 matrix, each row a vector of 0s except for the position of the label, which is 1.
def predict(X, Y, Theta1, Theta2):
    H = hypothesis(X, Theta1, Theta2)
    Y_predict = np.argmax(H, axis = 1)
    Y_test = np.argmax(Y, axis = 1)
    accuracy = sum(Y_predict == Y_test)/m_test * 100
    #accuracy = accuracy.item() #change to scalar
    print('Prediction accuracy = {:.2f}%'.format(accuracy))
    return Y_predict

#Make predictions on test set using Theta
X_test = df_to_X(iris_test)
Y_test = df_to_y(iris_test, species_names)

Y_predict = predict(X_test, Y_test, Theta1_trained, Theta2_trained)