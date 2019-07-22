#!/usr/bin/env python
# coding: utf-8

# In[63]:


import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt


# In[68]:


def eval_loss(w, b, x_list, gt_y_list):
    
     # loss function
    avg_loss_list = 0.5 * (w * x_list + b - gt_y_list) ** 2
       
    avg_loss = sum(avg_loss_list)/len(gt_y_list)
    return avg_loss


# In[109]:


def cal_step_gradient(batch_x_list, batch_gt_y_list, w, b, lr):
    avg_dw, avg_db = 0, 0
    batch_size = len(batch_x_list)
    #print(bat)
    
    pred_y_list = w * batch_x_list + b
    
    db_list = pred_y_list - batch_gt_y_list
    dw_list = db_list * batch_x_list
    
    avg_dw = sum(dw_list)/batch_size
    avg_db = sum(db_list)/batch_size
    
    w -= lr * avg_dw
    b -= lr * avg_db
    return w, b


# In[74]:


def gen_sample_data():
    num_samples = 100
    w = np.random.randint(0, 10) + np.random.rand()		# for noise random.random[0, 1)
    b = np.random.randint(0, 5) + np.random.rand()

    x_list = np.random.randint(0, 100, num_samples) * np.random.rand(num_samples)
    y_list = w * x_list + b + np.random.rand(num_samples) * np.random.randint(-1, 1, num_samples)

    return x_list, y_list, w, b


# In[115]:


def train(x_list, gt_y_list, batch_size, lr, max_iter):
    w = 0
    b = 0
    num_samples = len(x_list)
    
    for i in range(max_iter):
        batch_idxs = np.random.choice(len(x_list), batch_size)
    
        batch_x = np.take(x_list, batch_idxs)
        batch_y = np.take(gt_y_list, batch_idxs)
        
        w, b = cal_step_gradient(batch_x, batch_y, w, b, lr)
        print('w:{0}, b:{1}'.format(w, b))
        print('loss is {0}'.format(eval_loss(w, b, x_list, gt_y_list)))


# In[92]:


def run():
    x_list, y_list, w, b = gen_sample_data()
    lr = 0.001
    max_iter = 10000
    train(x_list, y_list, 50, lr, max_iter)


# In[117]:


run()


# In[ ]:




