#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 11:26:17 2025

@author: felipe
"""

# load digits
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import random
from scipy.linalg import svd
import matplotlib.pyplot as plt

def TC(X):
    """
    Transpose conjugated operation of a matrix

    Parameters
    ----------
    X : 2D complex array
        A matrix.

    Returns
    -------
    X* : 2D complex array
        Transpose conjugated of X.

    """
    
    return X.conj().T


#
def create_digit_image(digit, dx=0, dy=0, angle=0, font=0, noise=0):
    font_size=20
    img_size=(24,24)
    fonts=[ImageFont.load_default(size=24),ImageFont.truetype('Arial',size=font_size),
           ImageFont.truetype('Lato-Regular',size=font_size),
           ImageFont.truetype('DejaVuSans',size=font_size),
           ImageFont.truetype('comic',size=font_size),
           ImageFont.truetype('couri',size=font_size)]
    # Crear imagen en blanco
    img = Image.new('L', img_size, color=0)
    draw = ImageDraw.Draw(img)

    # Escribir el número centrado
    w, h = 10,22
    fontType=fonts[font]
    draw.text(((img_size[0]-w)/2+dx, (img_size[1]-h)/2+dy), str(digit), 255, font=fontType)

    # Rotar
    img = img.rotate(angle, resample=Image.BILINEAR)

    arr = np.array(img).astype(np.int16)
    if noise > 2 and noise < 10:
        #agregar ruido 1
        arr = np.array(img).astype(np.int16)
        arr[np.random.randint(0,img_size[0],(noise,2))]=np.uint8(0)
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if noise > 10:
        # Agregar ruido 2
        arr = np.array(img).astype(np.int16)
        arr += np.random.randint(-noise, noise + 1, arr.shape)
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr.flatten()

image_size=24*24
variants_per_digit=100
total_patterns=10*variants_per_digit
digit_matrix=np.zeros((total_patterns,image_size))
labels_vector=np.repeat(np.arange(0,10),variants_per_digit)
labels_matrix=np.zeros((total_patterns,10))
for l,label in enumerate(labels_vector):
    labels_matrix[l,label]=1

parent_dir='digits_dataset'
m=0
for digit in range(10):
    digit_dir = os.path.join(parent_dir, str(digit))
    for i in range(variants_per_digit):
        img=Image.open(os.path.join(digit_dir, f"{digit}_{i:03d}.png"))
        arr = np.array(img).astype(np.int16)
        digit_matrix[m,:]=arr.flatten()
        m+=1
theta=np.mean(digit_matrix)
digit_matrix=digit_matrix/theta

##Regresion
#X@W=labels
U,s,V=svd(digit_matrix)
rank=image_size
# La pseuodinversa de una matriz X=USV es es V* S^-1 U* (* transpuesta conjugada) 
W=TC(V[:rank,:])@np.diag(1/s[:rank])@TC(U[:,:rank])@labels_matrix

plt.figure()
result=digit_matrix@W
plt.subplot(1,2,1)
plt.plot(result)

## EML
# --- Kernel function (RBF) ---
def rbf_kernel(X1, X2, gamma):
    """
    Radial Basis Function (RBF) kernel.
    K(x, y) = exp(-gamma * ||x - y||^2)
    """
    X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)
    X2_sq = np.sum(X2**2, axis=1).reshape(1, -1)
    dist_sq = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
    return np.exp(-gamma * dist_sq)

def sigmoide(x):
    return 1/(1+np.exp(-x))

##EML
magic_factor=4.2
network_size=int(magic_factor*image_size)
Wrandom=np.random.randn(image_size,network_size)
bias=np.random.randn(network_size)
H=sigmoide(digit_matrix@Wrandom+bias)
lam=0.5
Beta=np.linalg.pinv(H.T@H+lam*np.eye(network_size))@H.T@labels_matrix

## Kernel EML
# Omega=rbf_kernel(digit_matrix,digit_matrix,gamma=0.05)
# C=1
# H=Omega+1/C*np.eye(total_patterns)
# Beta=np.linalg.pinv(H)@labels_matrix


result_EML=H@Beta
plt.subplot(1,2,2)
plt.plot(result_EML)

plt.savefig('train_result.png',dpi=200)

#%%
np.random.seed(3423)
plt.figure()

for digit in range(10):
    TP_regression=0
    TP_EML=0
    for mm in range(100):
        img_test=create_digit_image(digit, dx=0, dy=0, angle=np.random.randint(-2,2), font=np.random.randint(0,5), noise=np.random.randint(0,20))
        result_regression_test=(img_test/theta)@W
        
        H=sigmoide((img_test/theta)@Wrandom+bias)
        result_EML_test=H@Beta
            
        # Omega_test=rbf_kernel(digit_matrix, img_test.reshape([1,-1]), gamma=0.05)
        # result_KernelEML_test=Omega_test@Beta

        plt.plot(np.arange(10)-0.1,result_regression_test.T,'.',color='C0')
        plt.plot(np.arange(10)+0.1,result_EML_test.T,'.',color='C1') 
        out_regression_digit=np.argmax(result_regression_test.T)
        out_EML_digit=np.argmax(result_EML_test.T)
        plt.plot(out_regression_digit-0.1,result_regression_test[out_regression_digit],'k',marker='d')
        plt.plot(out_EML_digit+0.1,result_EML_test[out_EML_digit],'r',marker='d')
        
        if out_regression_digit==digit: 
            TP_regression+=1
        if out_EML_digit==digit: 
            TP_EML+=1
            
    print('accuraccies for ',digit,' Reg:' ,TP_regression,'% EML:',TP_EML)
plt.savefig('Test_results.png',dpi=300)

