#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob


# In[5]:


#folder = '/home/neeti/Documents/OCT2017/test/NORMAL/'
folder= "/home/neeti/Documents/OCT2017/val/NORMAL/*"
output_folder = '/home/neeti/Downloads/cropped/val/'
output_folder_amb =  '/home/neeti/Downloads/cropped/val_amb/'
i = 0


# In[ ]:


for filename in glob.glob(folder):
    #path = folder + filename
    #img = cv2.imread(path)
    i = i+1 
    img = cv2.imread(filename)
    img_name = filename.split('/')[-1]
    img = cv2.resize(img, (256, 256))
    cropped = img[40:236, 10:246]
    edges = cv2.Canny(cropped,200,300)
    ## find the non-zero min-max coords of canny
    pts = np.argwhere(edges>0)
    y1,x1 = pts.min(axis=0)
    y2,x2 = pts.max(axis=0)

    ## crop the region
    cropped_2 = cropped[y1:y2, x1:x2]
    
    if cropped_2.shape[0] <= 105:
        output_path = output_folder + img_name
    else:
        output_path = output_folder_amb + img_name
    cv2.imwrite(output_path, cropped_2)    
    if i%150 == 0:
        print('Images done:', i)


# In[ ]:




