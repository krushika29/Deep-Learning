
# coding: utf-8

# <h1><center>Deep Learning Home Work 2</center></h1>
# ## Team Members :-
# Krushika Tapedia (ktapedia) <br>
# Janvi Kothari (jkkothari) <br>

# In[2]:

import numpy as np
from matplotlib import pyplot as plt
import pdb

# In[57]:

train_faces = np.load('trainingFaces.npy')
train_labels = np.load('trainingLabels.npy')
test_faces = np.load('testingFaces.npy')
test_labels = np.load('testingLabels.npy')
pdb.set_trace()
#data_img = [train_faces[i].reshape(24,24) for i in range(len(train_faces))]
#train_labels = train_labels.reshape(2000,1)
#test_faces.shape


# In[4]:

#def display_face(img):
#    plt.imshow(img,cmap="gray",interpolation='nearest')
#    plt.show()


# In[5]:

w = np.random.randn(train_faces.shape[1])
#train_faces.shape[1]


# In[10]:

def J(w,x,y,alpha=0):
    wt = np.matrix.transpose(w)
    j = np.sum((np.dot(x,wt)-y)**2)
    return j/2


# In[17]:

def gradJ(w,x,y,alpha):
    try:
        wt,xt = np.matrix.transpose(w[:]), np.matrix.transpose(x[:])
        h = np.dot(x,wt)
        loss = h-y
        grad_j = np.dot(xt,loss)
        return grad_j
    except:
        print(xt.shape)


# ## Question 2 : Method 1
# Set Gradient to 0 and Solve

# In[24]:

def method1(train_faces,train_labels,test_faces,test_labels):
    # Computing 'w' -  weights
    w_transpose = np.matrix.transpose(np.dot(np.linalg.pinv(train_faces),train_labels))
    #Computing Testing Phase
    # y = w transpose * x
    #Computing the Cost in the training phase
    cost = J(w_transpose,train_faces,train_labels,0)
    print("Training Cost for Method 1 =",cost)
    cost_test = J(w_transpose,test_faces,test_labels,0)
    print("Testing Cost for Method 1 =",cost_test)


# In[25]:

method1(train_faces,train_labels,test_faces,test_labels)


# ## Question 2 : Method 2
# Gradient Descent - using epsilon (learning rate) and tolerance

# In[60]:

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
    return prev_cost,test_cost


# In[59]:

print("Training and Testing Cost for Method 2 = ",method2(train_faces,train_labels,test_faces,test_labels,0))


# In[54]:

def gradient_descent(w,x,y,alpha):
    a = np.dot(alpha,np.matrix.transpose(w))
    gradient = gradJ(w,x,y,alpha)+a
    return gradient


# ## Question 2 : Method 3
# Using Penalty alpha(alpha = 1000) and epsilon (learning rate = 1e-6)

# In[64]:

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
    print("Norm of W in Method 3",norm_w)
    return curr_cost,test_cost


# In[65]:

print("Training and Testing cost for Method 3 = ",method3(train_faces,train_labels,1e3))

