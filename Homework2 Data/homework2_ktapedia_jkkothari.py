# Deep Learning Home Work 2 
# Note:
# 1) Method 2 takes a little while to run, as the learning rate is low
# 2) You can also refer the results in the PDF !! The PDF has been generated from Jupyter Notebook !!

# ## Team Members :-
# Krushika Tapedia (ktapedia) <br>
# Janvi Kothari (jkkothari) <br>


import numpy as np
from matplotlib import pyplot as plt


train_faces = np.load('smile_data/trainingFaces.npy')
train_labels = np.load('smile_data/trainingLabels.npy')
test_faces = np.load('smile_data/testingFaces.npy')
test_labels = np.load('smile_data/testingLabels.npy')
train_labels = train_labels.reshape(2000,1)
test_labels = test_labels.reshape(test_labels.shape[0],1)


#This function computes the cost 'J' (equation is given in Method 1)
def J(w,x,y,alpha=0):
    wt = np.matrix.transpose(w)
    j = np.sum((np.dot(x,wt)-y)**2)
    return j/2

#This is a function that reports costs
def report_cost(w,alpha):
    print("Training Cost :",J(w,train_faces,train_labels,alpha),"\n")
    print("Testing Cost :",J(w,test_faces,test_labels,alpha),"\n")

#This is a function that would calculate the Gradient 
#of Cost function for Methods 1 and 2
def gradJ(w,x,y,alpha):
    try:
        wt,xt = np.matrix.transpose(w[:]), np.matrix.transpose(x[:])
        h = np.dot(x,wt)
        loss = h-y
        grad_j = np.dot(xt,loss)
        return grad_j
    except:
        print(xt.shape)


# This is the gradient descent function used in Method 3 which requires alpha
# and a derived from a new cost function
def gradient_descent(w,x,y,alpha):
    a = np.dot(alpha,np.matrix.transpose(w))
    gradient = gradJ(w,x,y,alpha)+a
    return gradient


# ## Question 2 : Method 1
# Set Gradient to 0 and Solve

def method1(train_faces,train_labels,test_faces,test_labels):
    # Computing 'w' -  weights
    w_transpose = np.matrix.transpose(np.dot(np.linalg.pinv(train_faces),train_labels))
    #Computing Testing Phase
    # y = w transpose * x
    return w_transpose


# ## Question 2 : Method 2
# Gradient Descent - using epsilon (learning rate) and tolerance

def method2(train_faces,train_labels,test_faces,test_labels,alpha):
    w = np.random.randn(1,train_faces.shape[1])
    tolerance = 10
    epsilon = 8e-6
    while tolerance > 0.001:
        prev_cost = J(w,train_faces,train_labels,alpha)
        prev_gradJ = gradJ(w,train_faces,train_labels,alpha)
        u = np.dot(prev_gradJ,epsilon)
        w = np.subtract(w,u.transpose())
        curr_cost = J(w,train_faces,train_labels,alpha)
        tolerance = np.absolute(prev_cost-curr_cost)
    test_cost = J(w,test_faces,test_labels,0)
    return w


# ## Question 2 : Method 3
# Using Penalty alpha(alpha = 1000) and epsilon (learning rate = 1e-6)

def method3(train_faces,train_labels,alpha=1e3):
    w = np.random.randn(1,train_faces.shape[1])
    tolerance = 10
    epsilon = 1e-6
    while tolerance > 0.001:
        penalty = (alpha/2)*np.dot(w,np.matrix.transpose(w))
        prev_cost = J(w,train_faces,train_labels,alpha)+penalty
        prev_gradJ = gradient_descent(w,train_faces,train_labels,alpha)
        u = np.dot(prev_gradJ,epsilon)
        w = np.subtract(w,u.transpose())
        curr_cost = J(w,train_faces,train_labels,alpha)+penalty
        tolerance = np.absolute(prev_cost-curr_cost)
    test_cost = J(w,test_faces,test_labels,1e3)
    norm_w = np.linalg.norm(w)
    return w


print("------Results for Method 1-----\n")
w1 = method1(train_faces,train_labels,test_faces,test_labels)
report_cost(w1,0)
print("------Results for Method 2-----\n")
w2 = method2(train_faces,train_labels,test_faces,test_labels,0)
report_cost(w2,0)
print("------Results for Method 3-----\n")
w3 =  method3(train_faces,train_labels,1e3)
report_cost(w3,1000)
print("----------------------------------")
print("Square Norm of W in Method 2",np.linalg.norm(w2)**2)
print("Square Norm of W in Method 3",np.linalg.norm(w3)**2)


# ### Output Description :
# 1) The above output shows the Testing and Training costs for all the three Methods <br>
# 2) Cost for Method 1 is obtained by directly setting the gradient zero and computing the weights <br>
# 3) We use learning rates like epsilon = 8e-6 for Method 2 and epsilon = 1e-6 for Method 3 <br>
# 4) If you compare method 2 and 3, training cost is higher in method 3, whereas testing cost is lower in Method 3 as compared to Method 2.<br>
# 5) The Output also has the Norm vales of W in methods 2 and 3
