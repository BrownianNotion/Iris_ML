import pandas as pd
import numpy as np 

iris_df = pd.read_csv("iris.csv")
#Three species of irises stored in list
species_names = iris_df['Species'].unique()
#shuffle rows
iris_shuffled_df = iris_df.sample(frac = 1)

"""
To change: add a function for shuffling and splitting.
"""
#SECTION 1: Split into training and test examples
m_train = 120
m_test = len(iris_df.index) - m_train
iris_train = iris_shuffled_df.head(m_train)
iris_test = iris_shuffled_df.tail(m_test)

#Convert dataframe to X matrix, adding column of 1s
#drops = list of column names to drop
def df_to_X(df, drops = ['Id', 'Species']):
    X = df.drop(drops, 1).to_numpy()
    m = X.shape[0]
    X = np.c_[np.ones(m), X]
    return X

"""
Rename as onehot and change if needed.
"""
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

"""
Possibly relegate code below under if name == "__main__"
"""
X_train = df_to_X(iris_train)
X_test = df_to_X(iris_test)
y_train = df_to_y(iris_train, species_names) #each column will be used in the multiclass classification
y_test = df_to_y(iris_test, species_names)

#SECTION 2: Calculate cost and minimise gradient
def sigmoid(z):
    return 1/(1 + np.exp(-z))

#computes the cost function for binary logistic regression
def cost_logistic(X, theta, y):
    m = y.shape[0] #no. of training examples
    h = sigmoid(X.dot(theta)) #hypothesis vector
    cost = -1/m * (y.T.dot(np.log(h)) + (1 - y).T.dot(np.log(1 - h))) # returns 1 by 1 array of cost
    return cost.item()


#SECTION 3: Perform gradient descent to optimise the gradient
def gradient_descent(X, y, theta0, alpha = 0.1): #alpha = learning rate
    max_iter = 500
    m = y.shape[0]
    theta = theta0
    for i in range(max_iter):
        theta = theta - alpha/m * (X.T).dot(sigmoid(X.dot(theta)) - y)
    return theta

n = 5 #number of parameters to use for regression (1 more than number of features due to bias unit)
theta0 = np.zeros([n, 1], dtype = int)

#matrix of column vectors for theta parameters. Column 1 is theta for setosa vs all, column 2 for veriscolor vs all etc.
k = len(species_names)
thetas = np.zeros((n, k))
for i in range(k):
    thetas[:, i] = gradient_descent(X_train, np.c_[y_train[:, i]], theta0)[:, 0]

#Make predictions
h_test = sigmoid(X_test.dot(thetas))
y_predicted = np.c_[np.argmax(h_test, axis = 1)]

#Prediction accuracy;
#vectorised change from 0,0,1 etc. to 0,1,2 (mx3 -> mx1 array)
y_test_labels = np.c_[np.argmax(y_test, axis = 1)]
accuracy = sum(y_predicted == y_test_labels)/m_test * 100
accuracy = accuracy.item() #change to scalar
print('{:.2f}%'.format(accuracy))
