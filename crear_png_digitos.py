#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 10:53:41 2025

@author: felipe
"""

# generate_digits_dataset.py
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import random

# Parámetros
img_size = (24, 24)
font_size = 20

output_dir = "digits_dataset"
num_variants_per_digit = 100
fonts=[ImageFont.load_default(size=24),ImageFont.truetype('Arial',size=font_size),
       ImageFont.truetype('Times_New_Roman',size=font_size),
       ImageFont.truetype('Verdana',size=font_size),
       ImageFont.truetype('Courier_New',size=font_size),
       ImageFont.truetype('Georgia',size=font_size)]
os.makedirs(output_dir, exist_ok=True)

def create_digit_image(digit, dx=0, dy=0, angle=0, font=0, noise=0):
    # Crear imagen en blanco
    img = Image.new('L', img_size, color=0)
    draw = ImageDraw.Draw(img)

    # Escribir el número centrado
    w, h = 10,22
    fontType=fonts[font]
    draw.text(((img_size[0]-w)/2+dx, (img_size[1]-h)/2+dy), str(digit), 255, font=fontType)

    # Rotar
    img = img.rotate(angle, resample=Image.BILINEAR)

    
    if noise > 2 and noise < 10:
        #agregar ruido 1
        arr = np.array(img).astype(np.int16)
        arr[np.random.randint(0,img_size[0],(noise,2))]=np.uint8(0)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    if noise > 10:
        # Agregar ruido 2
        arr = np.array(img).astype(np.int16)
        arr += np.random.randint(-noise, noise + 1, arr.shape)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    return img

# Generar dataset
for digit in range(10):
    digit_dir = os.path.join(output_dir, str(digit))
    os.makedirs(digit_dir, exist_ok=True)

    for i in range(num_variants_per_digit):
        dx = random.randint(-2, 2)
        dy = random.randint(-2, 2)
        angle = random.uniform(-15, 15)
        noise = random.randint(0, 20)
        font = random.randint(0,5)
        img = create_digit_image(digit, dx, dy, angle, font, noise)
        img.save(os.path.join(digit_dir, f"{digit}_{i:03d}.png"))