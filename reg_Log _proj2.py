#!/usr/bin/env python
# coding: utf-8

# In[61]:


import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math
get_ipython().run_line_magic('matplotlib', 'inline')


# In[62]:


# cell to load the dataset
x_train, y_train = load_data("data/ex2data2.txt")


# In[63]:


# cell to print the dataset of QA scores of two different test as 2D array (x_train) and the probability of acception as 1D array (y_train)  
# y_train = 1 ; if sensor is accepted
# y_train = 0 ; if sensor is rejected
print("x_train:", x_train[:5])
print("Type of x_train:",type(x_train))
print("y_train:", y_train[:5])
print("Type of y_train:",type(y_train))


# In[64]:


#cell to print the dimension of dataset
print ('The shape of x_train is: ' + str(x_train.shape))
print ('The shape of y_train is: ' + str(y_train.shape))
print ('We have m = %d training examples' % (len(y_train)))


# In[65]:


#cell to visualize the data graphically using matplotlib
plot_data(x_train, y_train[:], pos_label="Accepted", neg_label="Rejected")
plt.ylabel('Sensor Test 2') 
plt.xlabel('Sensor Test 1') 
plt.legend(loc="upper right")
plt.show()


# In[66]:


#cell to map the features of dataset (upto sixth power)
print("Original shape of data:", x_train.shape)
mapped_x =  map_feature(x_train[:, 0], x_train[:, 1])
print("Shape after feature mapping:", mapped_x.shape)
print("x_train[0]:", x_train[0])
print("mapped X_train[0]:", mapped_x[0])


# In[67]:


#cell to create the sigmoid function
def sigmoid(z):
    g = 1/(1 + np.exp(-z))
    return g


# In[71]:


#cell to create the cost function of logistic regression model
def cost_logistic(x, y, w, b, *argv):
    m,n = x.shape
    sum = 0
    for i in range (m):
        z_wb = 0
        for j in range (n):
            z_wb_ij = w[j]*x[i][j]
            z_wb += z_wb_ij
            
        z_wb += b
        f_wb = sigmoid(z_wb)
        sum += ( -y[i]*np.log(f_wb) - (1-y[i])*np.log(1-f_wb))
        
    total_cost = sum/m
    return total_cost


# In[72]:


#cell for computing the regularized cost of logistic regression model
def cost_logistic_reg(x, y, w, b, lambda_ = 1):
    m, n = x.shape
    cost_without_reg = cost_logistic(x, y, w, b) 
    reg_cost = 0.
    sum = 0
    
    for j in range(n):
        sum += w[j]**2
        
    reg_cost = lambda_ * sum / (2*m)
    total_cost = cost_without_reg + reg_cost
    return total_cost


# In[74]:


x_mapped = map_feature(x_train[:, 0], x_train[:, 1])
np.random.seed(1)
initial_w = np.random.rand(x_mapped.shape[1]) - 0.5
initial_b = 0.5
lambda_ = 0.5
cost = cost_logistic_reg(x_mapped, y_train, initial_w, initial_b, lambda_)

print("Regularized cost :", cost)
from public_tests import *
cost_logistic_reg_test(cost_logistic_reg)


# In[75]:


#cell for computing the gradient function for Logistic regression model
def gradient_logistic (x, y, w, b, *argv):
    m, n = x.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.
    for i in range(m):   
        z_wb = 0
        for j in range(n): 
            z_wb_ij = x[i, j] * w[j]
            z_wb += z_wb_ij
            
        z_wb += b
        f_wb = sigmoid(z_wb)
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        for j in range(n): 
            dj_dw_ij = (f_wb - y[i])*x[i][j]
            dj_dw[j] += dj_dw_ij
            
    dj_db = dj_db / m       
    dj_dw = dj_dw / m
    return dj_db, dj_dw 


# In[76]:


#cell for computing the gradient descent for logistic regression model
def gradient_descent_logistic(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_):
    m = len(x)
    J_history = []
    w_history = []
    
    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(x, y, w_in, b_in, lambda_)   
        w_in = w_in - alpha * dj_dw               
        b_in = b_in - alpha * dj_db              
       
        
        if i<100000:      
            cost =  cost_function(x, y, w_in, b_in, lambda_)
            J_history.append(cost)

        if i% math.ceil(num_iters/10) == 0 or i == (num_iters-1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w_in, b_in, J_history, w_history   


# In[77]:


# cell for computing the regularized gradient for logistic regression model
def gradient_logistic_reg(x, y, w, b, lambda_ = 1): 
    m, n = x.shape
    dj_db, dj_dw = compute_gradient(x, y, w, b)
    for j in range(n): 
        dj_dw_j_reg = (lambda_ / m) * w[j]
        dj_dw[j] = dj_dw[j] + dj_dw_j_reg

    return dj_db, dj_dw


# In[78]:


x_mapped = map_feature(x_train[:, 0], x_train[:, 1])
np.random.seed(1) 
initial_w  = np.random.rand(x_mapped.shape[1]) - 0.5 
initial_b = 0.5
lambda_ = 0.5
dj_db, dj_dw = gradient_logistic_reg(x_mapped, y_train, initial_w, initial_b, lambda_)

print(f"dj_db: {dj_db}", )
print(f"First few elements of regularized dj_dw:\n {dj_dw[:4].tolist()}", )
gradient_logistic_reg_test(gradient_logistic_reg)


# In[79]:


np.random.seed(1)
initial_w = np.random.rand(x_mapped.shape[1])-0.5
initial_b = 1.
lambda_ = 0.01    
iterations = 10000
alpha = 0.01

w,b, J_history,_ = gradient_descent_logistic(x_mapped, y_train, initial_w, initial_b, 
                                    cost_logistic_reg, gradient_logistic_reg, 
                                    alpha, iterations, lambda_)


# In[80]:


plot_decision_boundary(w, b, x_mapped, y_train)
plt.ylabel('Sensor Test 2') 
plt.xlabel('Sensor Test 1') 
plt.legend(loc="upper right")
plt.show()


# In[81]:


#cell for computing how well the trained model predicts
def predict(x, w, b):
    m, n = x.shape   
    p = np.zeros(m)
   
    z_wb = 0
    for i in range (m):
        for j in range (n):
            z_wb_ij = w[j]*x[i , j]
            z_wb += z_wb_ij
            
        z_wb += b
        f_wb = sigmoid(z_wb)
        p[i] = f_wb >= 0.5
        
    return p


# In[82]:


#cell for computing the accuracy of trained model
p = predict(x_mapped, w, b)
print('Train Accuracy: %f'%(np.mean(p == y_train) * 100))


# In[ ]:




