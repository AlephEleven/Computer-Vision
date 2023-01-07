import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
from math import ceil
from skimage.draw import line
from skimage.transform import resize
from scipy.ndimage import rotate

import sys
import os
import time

#@title HW1 Code: conv2d(img, kernel), gauss_filter(sig), compute_gradient(img, threshold), nonmax_supression(img_mag, img_dir)
#code from HW1 required for this assignment

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

def gauss_filter(sig, size=3):
  #gauss function (in parts)
  gauss_a = 1/(2*np.pi*sig**2)
  gauss_b = 2*sig**2
  gauss_2d = lambda x, y: gauss_a*np.e**(-(x**2+y**2)/gauss_b)

  #filter size should be std*3
  dist = int(sig*size)
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

#@title HW2 Code: nms_max(img), hessian_determinant(img)
def nms_max(img):
  #check 3x3 window, if center is max update img_nms, else continue
  row, col = img.shape
  img_nms = np.zeros_like(img)
  for i in range(1, row-1):
    for j in range(1, col-1):
      center = img[i][j]
      window = img[i-1:i+2, j-1:j+2]
      if center == np.max(window):
        img_nms[i][j] = np.max(window)

  return img_nms

def hessian_determinant(img):
  #apply derivates then do determinent
  K_sobel = np.array([[1,0,-1], [2,0,-2], [1,0,-1]])
  I_xx = conv2d(conv2d(img, K_sobel), K_sobel)
  I_yy = conv2d(conv2d(img, K_sobel.T), K_sobel.T)
  I_xy = conv2d(conv2d(img, K_sobel), K_sobel.T)

  hessian_det = I_xx * I_yy - (I_xy)**2

  return hessian_det

def img2coords(img):
  # flip bc imshow does y indexes 0 - 400 top to bottom.
  img = img
  row, col = img.shape
  return np.array([(j, i) for i in range(row) for j in range(col) if img[i][j] != 0.0])

# Part 1

class LinePreprocess:

  def __init__(self, gauss_std=2, rgb=True):
    self.std = gauss_std
    self.rgb = rgb

  def __call__(self, img):
    #normalize & convert to gray image
    if self.rgb:
      img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
      #normalize img between 0 - 255 bound (otherwise computing gradient will look awful)
      img_min, img_max = np.min(img), np.max(img)
      normalizer = np.vectorize(lambda x: (x - img_min)/(img_max - img_min) * (255.0 - 0.0) + 0.0)
      img = normalizer(img)

    img = img/255.0
    img = conv2d(img, gauss_filter(self.std))

    return img

def square_derivatives(img):
  #apply derivates
  K_sobel = np.array([[1,0,-1], [2,0,-2], [1,0,-1]])
  I_x = conv2d(img, K_sobel)
  I_y = conv2d(img, K_sobel.T)

  return I_x**2, I_y**2, I_x*I_y

class HarrisDetector:
  def __init__(self, window_size=3, alpha=0.05, gauss_std=1, max_features=1000):
    self.window_size = window_size
    self.alpha = alpha
    self.max_features = max_features
    self.gauss_std = 1

  def conv2d_harris(self, img, squares):
    '''
    Special case of 2d convolution, for dealing with three derivatives + gaussian
    '''
    gauss = gauss_filter(self.gauss_std)
    pad_width = ceil((gauss.shape[0]-1)/2)
    row, col = img.shape

    I_x, I_y, I_xy = squares

    corners = []

    #loop through non-padded part of image, and apply gaussian for NMS
    for i in range(pad_width, row-pad_width):
      for j in range(pad_width, col-pad_width):
        #get img window
        ind_row = (i-pad_width, i+1+pad_width)
        ind_col = (j-pad_width, j+1+pad_width)

        w_x = np.sum(I_x[ind_row[0]:ind_row[1], ind_col[0]:ind_col[1]] * gauss)
        w_y = np.sum(I_y[ind_row[0]:ind_row[1], ind_col[0]:ind_col[1]] * gauss)
        w_xy = np.sum(I_xy[ind_row[0]:ind_row[1], ind_col[0]:ind_col[1]] * gauss)

        M = np.array([[w_x, w_xy], [w_xy, w_y]])

        R = LA.det(M) - self.alpha*np.trace(M)**2
        corners.append([j, i, R])


    return np.array(corners)

  def img_corner(self, img, corners):
    corner_mat = np.zeros_like(img)
    for x, y, _ in corners:
      x = int(x)
      y = int(y)
      corner_mat[y, x] = 1

    return corner_mat
    

  def __call__(self, img_pp):
    '''
    Applies Harris Detector algorithm
    '''
    I_x, I_y, I_xy = square_derivatives(img_pp)

    corners = self.conv2d_harris(img_pp, [I_x, I_y, I_xy])

    #Find Corners via R > 0, sort by strongest R value, and get #max_features corners
    corners = corners[corners[:, 2] > 0]
    corners = corners[np.argsort(-corners[:, 2])][:self.max_features]

    #Reduce clustering via NMS
    img_corners = self.img_corner(img_pp, corners)
    img_corners = nms_max(img_corners)

    return corners, img_corners

  def plot_fig(self, corners, img):
    img_harris = img.copy()
    for x, y, _ in corners:
      x = int(x)
      y = int(y)
      img_harris[y-1:y+2, x-1:x+2] = [255, 0, 0]

    return img_harris
  
  def plot_fig2(self, corners, img):
    img_harris = img.copy()
    for x in range(corners.shape[0]):
      for y in range(corners.shape[1]):
        if corners[x,y] != 0.0:
          img_harris[x-1:x+2, y-1:y+2] = [255, 0, 0]

    return img_harris

class PatchSimilarity:
  def __init__(self, pt_window=12):
    self.pt_window=pt_window

  def NCC(self, imgs_data):
    '''
    Apply patch similarity using NCC metric and returns all matches (including the NCC rank)
    '''
    pad_width = int(self.pt_window/2)

    imgA_corners, imgA_gray = imgs_data[0]
    imgB_corners, imgB_gray = imgs_data[1]

    cornersA = img2coords(imgA_corners)
    cornersB = img2coords(imgB_corners)

    matches = []

    for cA_y, cA_x in cornersA:
      for cB_y, cB_x in cornersB:

        cA_window = imgA_gray[cA_x-pad_width:cA_x+1+pad_width, cA_y-pad_width:cA_y+1+pad_width]
        cB_window = imgB_gray[cB_x-pad_width:cB_x+1+pad_width, cB_y-pad_width:cB_y+1+pad_width]

        cA_norm = cA_window - np.mean(cA_window)
        cB_norm = cB_window - np.mean(cB_window)

        NCC = (np.sum(cA_norm * cB_norm))/(np.sqrt( np.sum(cA_norm**2) * np.sum(cB_norm**2) ))

        matches.append([NCC, cA_x, cA_y, cB_x, cB_y])

    matches = np.array(matches)

    return matches

  def plot_fig(self, imgA, imgB, matches):
    A_row, A_col = imgA.shape[0:2]
    img_patch = np.hstack((imgA, imgB))
    for _, a_x, a_y, b_x, b_y in matches:
      a_x, a_y, b_x, b_y = int(a_x), int(a_y), int(b_x), int(b_y)
      img_patch[a_x-2:a_x+3, a_y-2:a_y+3] = [255, 0, 0]
      img_patch[b_x-2:b_x+3, b_y-2+A_col:b_y+3+A_col] = [255, 0, 0]


      rr, cc = line(int(a_x), int(a_y), int(b_x), int(b_y+A_col))
      img_patch[rr, cc] = [0, 255, 0]

    return img_patch

# Part 2

class ImageAlign:
  def __init__(self, thd=0.2, p=0.9):
    self.M = np.zeros((3, 3))
    self.thd = thd
    self.p = p

  def lstq_affine(self, points):
    '''
    Build affine transformation using three points then apply least squares
    '''
    T = np.zeros((6,6))
    xt = np.zeros((6,1))
    ind = 0
    for A_x, A_y, B_x, B_y in points:
      T[ind*2, 0:2] = [A_x, A_y]
      T[ind*2+1, 2:4] = [A_x, A_y]
      T[ind*2, 4] = 1
      T[ind*2+1, 5] = 1

      xt[ind*2] = B_x
      xt[ind*2+1] = B_y

      ind += 1

    lq = LA.lstsq(T, xt, rcond=None)[0]

    M = lq[0:4, :]
    M = M.reshape((2,2))
    t = lq[4:6, :]

    return M, t

  def transform_point(self, point, M, t):
    return (M @ point.reshape((2,1)) + t).T

  def transform_align(self, point):
    return self.M @ np.hstack((point, np.array([1])))

  def find_inliers(self, matches, M, t):
    transforms = np.array([self.transform_point(match, M, t) for match in matches[:, 0:2]])
    transforms = np.reshape(transforms, (len(matches), 2))
    diff = (matches[:, 2:4] - transforms)**2
    diff = np.sqrt(diff[:, 0] + diff[:, 1])
    inliers = matches[diff < self.thd] 

    return inliers, diff



  def fit(self, points, img_shift, debug=False):
    '''
    Applies RAMSAC to find best affine transformation of matches
    '''
    matches = points.copy()
    matches = matches[:, 1:5]
    matches[:, 3] += img_shift
    max_inliers = 0

    sample_count = 0
    N = 1
    
    while(N > sample_count):
      rand_matches = matches[np.sort(np.random.randint(len(matches), size=3))]
      M, t = self.lstq_affine(rand_matches)

      inliers, diff = self.find_inliers(matches, M, t)
      if len(inliers) > max_inliers:
        max_inliers = len(inliers)
        self.M[0:2, 0:2] = M
        self.M[0:2, 2] = t.flatten()
        self.M[2, 2] = 1

      e = 1 - len(inliers)/len(matches)
      N = np.log(1-self.p)/np.log(1 - (1 - e)**3)
      #there tends to be an issue with no inliers being greater than sample count, so set N>sample_count in this case
      if len(inliers) == 0:
        N = sample_count+2
      sample_count += 1

    avg_error = 1/(len(diff))*np.sum(diff)
    if debug:
      print(f"RANSAC iterations: {sample_count}")
      print(f"Inlier Count: {len(inliers)}: ")
      print("y \t| x \t| yt \t| xt")
      print(inliers)
      print(f"Avg. Reprojection Error: {avg_error}")

    return inliers, sample_count, avg_error

  def avg_rgb(self, rgbA, rgbB):
    #ref: https://stackoverflow.com/questions/649454/what-is-the-best-way-to-average-two-colors-that-define-a-linear-gradient
    return [np.sqrt((rgbA[0]**2 + rgbB[0]**2)/2), np.sqrt((rgbA[1]**2 + rgbB[1]**2)/2), np.sqrt((rgbA[1]**2 + rgbB[1]**2)/2)]

  def transform(self, imgA, imgB):
    A_row, A_col = imgA.shape[0:2]
    img_transform = np.hstack((imgA, imgB))

    #pad edges of transform to fit the max transformed top/bottom corners
    transform_corners = np.array([self.transform_align(np.array(corner))[0:2] for corner in [[0,0], [A_row, 0], [0, A_col], [A_row, A_col]  ]])
    edge = ceil(np.max([ np.abs(np.min(transform_corners[:, 0])), np.abs(np.max(transform_corners[:, 0]) - A_row)]))

    img_transform = np.pad(img_transform, pad_width=[(edge, edge),(0, 0),(0, 0)], mode='constant')



    leftmost_pt = A_col

    for row in range(edge, A_row+edge):
      for col in range(A_col):
        transform_pt = self.transform_align(np.array([row, col]))[0:2]
        pt_x, pt_y = round(transform_pt[0]), round(transform_pt[1])
        img_transform[row, col] = [0,0,0]
        if not np.array_equal(img_transform[pt_x, pt_y], [0,0,0]):
          img_transform[pt_x, pt_y] = self.avg_rgb(imgA[row-edge, col], img_transform[pt_x, pt_y])
        else:
          img_transform[pt_x, pt_y] = imgA[row-edge, col]


    #chop off excess columns from left
    for col in range(img_transform.shape[1]):
      col_ind = img_transform[:, col]
      col_ind = col_ind[col_ind != 0]
      if col_ind.size == 0:
        leftmost_pt = col

    img_transform = img_transform[:, leftmost_pt:]

    return img_transform

def rotate_image(img, angle = 45):
  print(f'Rotating Image {angle} Degrees..')
  rotated = rotate(img, angle, reshape=False)
  rotated = resize(rotated, (img.shape[0], img.shape[1]))
  rotated = (rotated * 255).astype(np.uint8)
  return rotated

if __name__ == "__main__":
  args = sys.argv[1:]
  if len(args) > 0:
    try:
      if args[0]=="p":
        args[1] = 1000
        args[2] = 12
        args[3] = 20
      else:
          args[1] = int(args[1])
          args[2] = int(args[2])
          args[3] = int(args[3])
    except:
        raise Exception("Usage: [python/python3] TwoViewAlignment.py [p/np: default/custom] (int: max_features) (int: similiarity_window) (int: correspondence) (str: image A path) (str: image B path)") 
    max_fts, sim_window, num_matches, *fps = args[1:]
  else:
    max_fts, sim_window, num_matches, fps = 1000, 12, 20, ["data/uttower_left.jpg", "data/uttower_right.jpg"]

  print(f"Running two view alignment with:\n std: {max_fts}\n similarity window: {sim_window}\n correspondences: {num_matches}\n files: {fps}")
  imgs = [plt.imread(fp) for fp in fps]
  # imgs[0] = rotate_image(imgs[0], 45) FOR ROTATION
  print(f"Read images: {fps}")

  dirname = f"tva-results-{int(time.time())}"
  os.mkdir(dirname)

  print("Preprocessing images...")
  lpp = LinePreprocess()
  imgs_gray = [lpp(img) for img in imgs]

  print(f"Finding best {max_fts} corners")
  harris = HarrisDetector(max_features=max_fts)
  corners = [harris(img_gray) for img_gray in imgs_gray]
  imgs_corners = [corners[i][1] for i in range(len(corners))]

  plt.imsave(f"{dirname}/left_harris.png",harris.plot_fig2(imgs_corners[0], imgs[0]), format='png')
  plt.imsave(f"{dirname}/right_harris.png",harris.plot_fig2(imgs_corners[1], imgs[1]), format='png')

  print(f"Finding {num_matches} correspondences of top {max_fts} features")
  ps = PatchSimilarity(sim_window)
  imgs_data = [[imgs_corners[i], imgs_gray[i]] for i in range(len(imgs_gray))]
  matches = ps.NCC(imgs_data)

  top_n = num_matches
  best_matches = matches[np.argsort(-matches[:, 0])]
  plt.imsave(f"{dirname}/top_{num_matches}_correspondences.png",ps.plot_fig(imgs[0], imgs[1], best_matches[:top_n]), format='png')

  random_n = 30
  random_matches = top_n+np.random.choice(len(matches)-top_n, random_n, replace=False)
  align_matches = np.vstack((best_matches[:top_n], matches[random_matches]))
  plt.imsave(f"{dirname}/top_{num_matches}_random_30_correspondences.png",ps.plot_fig(imgs[0], imgs[1], align_matches), format='png')

  print("Aligning images...")
  ia = ImageAlign()
  img_shift = imgs[0].shape[1]
  fit_matches = ia.fit(best_matches[:20], img_shift, debug=True)
  img_align = ia.transform(imgs[0], imgs[1])
  #plt.imshow(img_align) top n align
  plt.imsave(f"{dirname}/top_{num_matches}_align.png",img_align, format='png')
  print(f"Processed Top-{num_matches} Alignment")

  ia2 = ImageAlign()
  img_shift = imgs[0].shape[1]
  fit_matches = ia2.fit(align_matches, img_shift, debug=True)
  img_align = ia2.transform(imgs[0], imgs[1])
  plt.imsave(f"{dirname}/top_{num_matches}_random_30_align.png",img_align, format='png')
  print(f"Processed Top-{num_matches} and random 30 Alignment")




  