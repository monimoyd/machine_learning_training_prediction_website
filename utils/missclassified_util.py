
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# In[ ]:


def imshow2(img):
    #img = img / 2 + 0.5     # unnormalize
    #npimg = img.numpy()
    plt.imshow(np.transpose(img, (1,2, 0)))


# In[ ]:


def plot_images(miss_class,miss_class_lbl,miss_class_pr_lbl,classes):
  num_display = 10
  plt.figure(figsize=(30,8))
  columns = 10
  i= 0
      # Display the list of 25 misclassified images
  for index, image in enumerate(miss_class) :
    ax = plt.subplot(1, 10, i+1)
    #ax=fig.add_subplot(1,number_of_files,i+1)
    ax.set_title("Actual: " + str(classes[int(miss_class_lbl[index])]) + "\n"+", Predicted: " + str(classes[int(miss_class_pr_lbl[index])]))
    ax.axis('off')
    #plt.imshow(image.cpu().numpy())
    imshow2(image.cpu().numpy())
    i +=1
    if i==num_display:
      break


# In[ ]:


def plot_misclassified(model,test_dl,device,classes):
  miss_class_images = []
  miss_labels = []
  miss_pr_labels = []
  dataiter = iter(test_dl)
  while True:
    try:
      data1 = dataiter.next()
      print(data1['X'].shape)
      with torch.no_grad():
        images=[]
        labels=[]
        images, labels = data1['X'].to(device), data1['Y'].to(device)
        outputs = model(images)
        predicted = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        labels_nm = labels.cpu().numpy()
        predicted_nm = predicted.cpu().numpy()
        img_nm = images.cpu().numpy()
        img_list = [img_nm[i] for i in range(img_nm.shape[0])]
      #print(predicted_nm.tolist())
      #print(labels_nm.tolist())
        pr_list=predicted_nm.tolist()
        lb_list =labels_nm.tolist()
      #img_list=img_nm.tolist()
      #print(len(pr_list))
      #print(len(lb_list))
        for i in range(len(lb_list)):
          if lb_list[i] != pr_list[i][0]:
            #print(i)
            miss_class_images.append(images[i])
            miss_labels.extend(str(lb_list[i]))
            miss_pr_labels.extend(str(pr_list[i][0]))
        #print(imshow1(images[i].cpu().numpy()))

    except:
      print("Oops! Error occurred.")
      break
  #return miss_class_images,miss_labels,miss_pr_labels
  miss_class_1 = []
  miss_class_1_lbl= []
  miss_class_1_pr_lbl = []
  miss_class_2 = []
  miss_class_2_lbl= []
  miss_class_2_pr_lbl = []
  miss_class_3 = []
  miss_class_3_lbl= []
  miss_class_3_pr_lbl = []
  miss_class_4 = []
  miss_class_4_lbl= []
  miss_class_4_pr_lbl = []
  for i in range(len(miss_labels)):
  #print(miss_labels[i])
    if int(miss_labels[i]) == 0:
    #print("Got 0")
      miss_class_1.append(miss_class_images[i])
      miss_class_1_lbl.append(miss_labels[i])
      miss_class_1_pr_lbl.append(miss_pr_labels[i])
    elif int(miss_labels[i]) == 1:
    #print("Got1")
      miss_class_2.append(miss_class_images[i])
      miss_class_2_lbl.append(miss_labels[i])
      miss_class_2_pr_lbl.append(miss_pr_labels[i])
    elif int(miss_labels[i]) == 2:
    #print("Got2")
      miss_class_3.append(miss_class_images[i])
      miss_class_3_lbl.append(miss_labels[i])
      miss_class_3_pr_lbl.append(miss_pr_labels[i])
    elif int(miss_labels[i]) == 3:
    #print("Got3")
      miss_class_4.append(miss_class_images[i])
      miss_class_4_lbl.append(miss_labels[i]) 
      miss_class_4_pr_lbl.append(miss_pr_labels[i])
  plot_images(miss_class_1,miss_class_1_lbl,miss_class_1_pr_lbl,classes)
  plot_images(miss_class_2,miss_class_2_lbl,miss_class_2_pr_lbl,classes)
  plot_images(miss_class_3,miss_class_3_lbl,miss_class_3_pr_lbl,classes)
  plot_images(miss_class_4,miss_class_4_lbl,miss_class_4_pr_lbl,classes)                     

