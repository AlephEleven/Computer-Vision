import matplotlib.pyplot as plt
import numpy as np
from math import ceil

import sys
import time
import os

def conv2d(img, kernel):
  #pad width for nxn kernel
  pad_width = ceil((kernel.shape[0]-1)/2)

  #use edge padding
  img_padded = np.pad(img, pad_width, mode="edge")
  img_conv = np.zeros_like(img_padded)
  
  row, col = img.shape
  row_conv, col_conv = img_conv.shape

  #loop through non-padded part of image
  for i in range(pad_width, row+pad_width):
    for j in range(pad_width, col+pad_width):
      #get img window
      img_window = img_padded[i-pad_width:i+1+pad_width, j-pad_width:j+1+pad_width]
      #conv_ij = sum(window_ij*kernel*ij)
      img_conv[i,j] = np.sum(img_window*kernel)

  #drop padding
  return img_conv[pad_width:row_conv-pad_width, pad_width:col_conv-pad_width]

def gauss_filter(sig):
  #gauss function (in parts)
  gauss_a = 1/(2*np.pi*sig**2)
  gauss_b = 2*sig**2
  gauss_2d = lambda x, y: gauss_a*np.e**(-(x**2+y**2)/gauss_b)

  #filter size should be std*3
  dist = int(sig*3)
  grad = range(-dist, dist+1, 1)

  #generate gauss map
  return np.array([[gauss_2d(x, y) for y in grad] for x in grad])

def compute_gradient(img, sig):
  #apply sobel
  K_sobel = np.array([[1,0,-1], [2,0,-2], [1,0,-1]])
  img_gx = conv2d(img, K_sobel)
  img_gy = conv2d(img, K_sobel.T)

  row, col = img.shape
  mag, dir = np.zeros_like(img), np.zeros_like(img)

  for i in range(row):
    for j in range(col):
      #grad. magnitude
      gm_ij = (img_gx[i][j]**2 + img_gy[i][j]**2)**0.5

      #grad. direction (in radii)
      gd_ij = (np.arctan2(img_gy[i][j],img_gx[i][j]) * 180) / np.pi

      #if mag > threshold, update mag and dir at ij
      if gm_ij > sig:
        mag[i][j] = gm_ij
        dir[i][j] = gd_ij

  return mag, dir

def nonmax_supression(img_mag, img_dir):

  img_nms = np.zeros_like(img_mag)
  row, col = img_mag.shape

  #loop through array, with spacing for edges
  for i in range(1, row-1):
    for j in range(1, col-1):
      angle = img_dir[i][j]

      #if values inside angle bound for both directions are greater than magnitude: use value
      #else: set to 0

      #horizontal: -22.5:22.5, -157.5:-180 & 157.5:180
      if (angle >= -22.5 and angle <= 22.5) or (angle <= -157.5 and angle >= -180) or (angle >= 157.5 and angle <= 180):
        if(img_mag[i][j] >= img_mag[i][j+1] and img_mag[i][j] >= img_mag[i][j-1]):
          img_nms[i][j] = img_mag[i][j]
        else:
          img_nms[i][j] = 0

      #diagonal 1 22.5:67.5, -112.5:-157.5
      elif (angle >= 22.5 and angle <= 67.5) or (angle <= -112.5 and angle >= -157.5):
        if(img_mag[i][j] >= img_mag[i+1][j+1] and img_mag[i][j] >= img_mag[i-1][j-1]):
          img_nms[i][j] = img_mag[i][j]
        else:
          img_nms[i][j] = 0
      #vertical 67.5:112.5, -67.5:-112.5
      elif (angle >= 67.5 and angle <= 112.5) or (angle <= -67.5 and angle >= -112.5):
        if(img_mag[i][j] >= img_mag[i+1][j] and img_mag[i][j] >= img_mag[i-1][j]):
          img_nms[i][j] = img_mag[i][j]
        else:
          img_nms[i][j] = 0

      #diagonal 2 112.5:157.5, -22.5:-67.5
      elif (angle >= 112.5 and angle <= 157.5) or (angle <= -22.5 and angle >= -67.5):
        if(img_mag[i][j] >= img_mag[i+1][j-1] and img_mag[i][j] >= img_mag[i-1][j+1]):
          img_nms[i][j] = img_mag[i][j]
        else:
          img_nms[i][j] = 0

  return img_nms

class EdgeDetection:

  def __init__(self, gauss_std, edge_threshold):
    self.std = gauss_std
    self.threshold = edge_threshold

  def forward(self, img):
    #normalize image
    img = img/255.0
    img = conv2d(img, gauss_filter(self.std))
    #normalize threshold as well
    img_mag, img_dir = compute_gradient(img, self.threshold/255.0)
    img = nonmax_supression(img_mag, img_dir)
    #convert non-zeros to 1 to highlight results
    img[img > 0] = 1


    return img



if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) > 0:
      try:
        if args[0]=="p":
          args[1] = 2
          args[2] = 30
        else:
            args[1] = float(args[1])
            args[2] = float(args[2])
      except:
          raise Exception("Usage: [python/python3] EdgeDetection.py [p/np: default/custom] (float: gaussian std) (float: gradient threshold) (str: image path/s)+") 
      std, thd, *fps = args[1:]
    else:
      std, thd, fps = 2, 30, ["data/plane.pgm"]

    print(f"Running edge detection with:\n std: {std}\n threshold: {thd}\n files: {fps}")
    ed = EdgeDetection(std, thd)
    imgs = [plt.imread(fp) for fp in fps]

    #create folder of results
    dirname = f"ed-results-{int(time.time())}"
    os.mkdir(dirname)

    #apply edge detection and save to folder
    for ind, img in enumerate(imgs):
      img_edge = ed.forward(img)
      plt.imsave(f"{dirname}/{fps[ind].split('/')[-1].split('.')[0]}-{int(time.time())}.png", img_edge, format="png", cmap="gray")
      print(f"Processed {fps[ind]}:\t{round(100*(ind+1)/len(fps), 3)}%")