# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image, ImageTk
from scipy import ndimage
from lr_utils import load_dataset

#----------------------Load and clean the dataset----------------------------------------

#Load the data (cat/not cat datasets)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

#Lets visualize the train set
print(train_set_x_orig)
print(train_set_y)
print(classes)

#Example of image to test and see if the import worked
#index = 20
#plt.imshow(train_set_x_orig[index])
#print("y = " + str(train_set_y[:,index]) + ", it's a '" + classes[np.squeeze(train_set_y[:,index])].decode("utf-8") + "' picture.")

#Lets get some basic data about our image numpy arrays
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

print("Number of training examples: m_train = " + str(m_train))
print("Number of test examples: m_test = " + str(m_test))
print("Height/Width of each image: num_px = " + str(num_px))
print("Each image is of size: ("+ str(num_px) + ", " + str(num_px) + ", 3)")
print("train_set_x shape: " + str(train_set_x_orig.shape))
print("train_set_y shape: " + str(train_set_y.shape))
print("test_set_x shape : " + str(test_set_x_orig.shape))
print("test_set_y shape: "+ str(test_set_y.shape))

#Will now flatten the numpy array from (num_px, num_px, 3) to (num_px*num_px*3, 1) 
#this will make it easier for us so that each image in one numpy array column
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

print("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print("train_set_y shape: " + str(train_set_y.shape))
print("test_set_x_flatten shape: "+ str(test_set_x_flatten.shape))
print("test_set_y shape: "+ str(test_set_y.shape))

#Standardize the dataset for images by dividing each by 255
train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

#--------------------------Build the Logisitic Regression framework-------------------------

#We will be using a sigmoid function for our Activation, in Neural Networks most are not ReLU due to speed of calc

def sigmoid(z):
    s = 1/(1+np.exp(-(z)))
    return s

#Create function to set both w and b to 0 to start with
def initialize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0
    return w,b

#Create a function that calculates the current SSE
def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X)+b)
    cost = -1/m * np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))
    dw = (1/m) * (np.dot(X,(A-Y).T))
    db = (1/m) * (np.sum(A-Y))
    
    cost = np.squeeze(cost)
    
    grads = {"dw": dw, "db": db}
    
    return grads, cost

#Create a function that moves the estimates around and calculates the SSE to find optimal w and b
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w,b,X,Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w-learning_rate*dw
        b = b-learning_rate*db
        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w":w,"b":b}
    grads = {"dw":dw,"db":db}
    
    return params, grads, costs

def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)
    A = sigmoid(np.dot(w.T,X)+b)
    for i in range(A.shape[1]):
        if A[0,i] <= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1
            
    return Y_prediction

#----------------------Merge all the components in to a model----------------------------

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    global D
    #Initialize paramters with 0
    w,b = initialize_with_zeros(X_train.shape[0])
    #Perform Gradient Descent
    parameters, grads, costs = optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost)
    #Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    #Predict test/train set examples
    Y_prediction_test = predict(w,b,X_test)
    Y_prediction_train = predict(w,b,X_train)
    #Print train/test errors
    print("train accuracy: {} %".format(100-np.mean(np.abs(Y_prediction_train-Y_train))*100))
    print("test accuracy: {} %".format(100-np.mean(np.abs(Y_prediction_test-Y_test))*100))
    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w":w,
         "b":b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}
    D = d


#----------------------Test on our own image----------------------------------------

def run_on_own_image(my_image):
    fname = my_image
    image = np.array(ndimage.imread(fname,flatten=False))
    my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1,num_px*num_px*3)).T
    my_predicted_image = predict(D["w"], D["b"],my_image)
    
    #plt.imshow(image)
    print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicted a \"" 
      + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") + "\" picture.")
    labelText = tk.Label(text="Your algorithm predicted a " + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8"))
    labelText.text = "Your algorithm predicted a " + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8")
    labelText.pack()    
    img = Image.open(fname)
    img = img.resize((300,300), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(img)
    label = tk.Label(image=photo)
    label.image = photo
    label.pack()

def choose_file():
    global fileNameGlobal
    filename = askopenfilename()
    fileNameGlobal = filename
#--------------------------Run the model------------------------------------------------

#d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)
import tkinter as tk
from tkinter.filedialog import askopenfilename

master = tk.Tk()

trainButton = tk.Button(master, text="Train Model", command = lambda: model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True))
trainButton.pack()

fileButton = tk.Button(master, text="Choose your own file", command=choose_file)
fileButton.pack()

ownImageButton = tk.Button(master, text="Test Own Image", command = lambda: run_on_own_image(fileNameGlobal))
ownImageButton.pack()

tk.mainloop()


