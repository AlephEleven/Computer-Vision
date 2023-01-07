import numpy as np
from math import ceil
from skimage.draw import line
import cv2
import traceback
from typing import Union
from collections import defaultdict

import sys
import time
import os

def conv2d(img, kernel):
    # pad width for nxn kernel
    pad_width = ceil((kernel.shape[0] - 1) / 2)

    # use edge padding
    img_padded = np.pad(img, pad_width, mode="edge")
    img_conv = np.zeros_like(img_padded)

    row, col = img.shape
    row_conv, col_conv = img_conv.shape

    # loop through non-padded part of image
    for i in range(pad_width, row + pad_width):
        for j in range(pad_width, col + pad_width):
            # get img window
            img_window = img_padded[i - pad_width:i + 1 + pad_width, j - pad_width:j + 1 + pad_width]
            # conv_ij = sum(window_ij*kernel*ij)
            img_conv[i, j] = np.sum(img_window * kernel)

    # drop padding
    return img_conv[pad_width:row_conv - pad_width, pad_width:col_conv - pad_width]

def gauss_filter(sig, size=3):
    # gauss function (in parts)
    gauss_a = 1 / (2 * np.pi * sig ** 2)
    gauss_b = 2 * sig ** 2
    gauss_2d = lambda x, y: gauss_a * np.e ** (-(x ** 2 + y ** 2) / gauss_b)

    # filter size should be std*3
    dist = int(sig * size)
    grad = range(-dist, dist + 1, 1)

    # generate gauss map
    return np.array([[gauss_2d(x, y) for y in grad] for x in grad])

def compute_gradient(img, sig):
    # apply sobel
    K_sobel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    img_gx = conv2d(img, K_sobel)
    img_gy = conv2d(img, K_sobel.T)

    row, col = img.shape
    mag, dir = np.zeros_like(img), np.zeros_like(img)

    for i in range(row):
        for j in range(col):
            # grad. magnitude
            gm_ij = (img_gx[i][j] ** 2 + img_gy[i][j] ** 2) ** 0.5

            # grad. direction (in radii)
            gd_ij = (np.arctan2(img_gy[i][j], img_gx[i][j]) * 180) / np.pi

            # if mag > threshold, update mag and dir at ij
            if gm_ij > sig:
                mag[i][j] = gm_ij
                dir[i][j] = gd_ij

    return mag, dir

def nonmax_supression(img_mag, img_dir):
    img_nms = np.zeros_like(img_mag)
    row, col = img_mag.shape

    # loop through array, with spacing for edges
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            angle = img_dir[i][j]

            # if values inside angle bound for both directions are greater than magnitude: use value
            # else: set to 0

            # horizontal: -22.5:22.5, -157.5:-180 & 157.5:180
            if (angle >= -22.5 and angle <= 22.5) or (angle <= -157.5 and angle >= -180) or (
                    angle >= 157.5 and angle <= 180):
                if (img_mag[i][j] >= img_mag[i][j + 1] and img_mag[i][j] >= img_mag[i][j - 1]):
                    img_nms[i][j] = img_mag[i][j]
                else:
                    img_nms[i][j] = 0

            # diagonal 1 22.5:67.5, -112.5:-157.5
            elif (angle >= 22.5 and angle <= 67.5) or (angle <= -112.5 and angle >= -157.5):
                if (img_mag[i][j] >= img_mag[i + 1][j + 1] and img_mag[i][j] >= img_mag[i - 1][j - 1]):
                    img_nms[i][j] = img_mag[i][j]
                else:
                    img_nms[i][j] = 0
            # vertical 67.5:112.5, -67.5:-112.5
            elif (angle >= 67.5 and angle <= 112.5) or (angle <= -67.5 and angle >= -112.5):
                if (img_mag[i][j] >= img_mag[i + 1][j] and img_mag[i][j] >= img_mag[i - 1][j]):
                    img_nms[i][j] = img_mag[i][j]
                else:
                    img_nms[i][j] = 0

            # diagonal 2 112.5:157.5, -22.5:-67.5
            elif (angle >= 112.5 and angle <= 157.5) or (angle <= -22.5 and angle >= -67.5):
                if (img_mag[i][j] >= img_mag[i + 1][j - 1] and img_mag[i][j] >= img_mag[i - 1][j + 1]):
                    img_nms[i][j] = img_mag[i][j]
                else:
                    img_nms[i][j] = 0

    return img_nms

def nms_max(img):
    # check 3x3 window, if center is max update img_nms, else continue
    row, col = img.shape
    for i in range(1, row - 1):
        for j in range(1, row - 1):
            center = img[i][j]
            window = img[i - 1:i + 2, j - 1:j + 2]
            if center == np.max(window):
                img[i][j] = np.max(window)

    return img

# 1)
def hessian_determinant(img):
    # apply derivates then do determinent
    K_sobel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    I_xx = conv2d(conv2d(img, K_sobel), K_sobel)
    I_yy = conv2d(conv2d(img, K_sobel.T), K_sobel.T)
    I_xy = conv2d(conv2d(img, K_sobel), K_sobel.T)

    hessian_det = I_xx * I_yy - (I_xy) ** 2

    return hessian_det

class LinePreprocess:

    def __init__(self, h_thd, nm_thd, gauss_std=2, rgb=True):
        self.std = gauss_std
        self.h_thd = h_thd
        self.nm_thd = nm_thd
        self.rgb = rgb

    def __call__(self, img):
        # normalize & convert to gray image
        if self.rgb:
            img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
            # normalize img between 0 - 255 bound (otherwise computing gradient will look awful)
            img_min, img_max = np.min(img), np.max(img)
            normalizer = np.vectorize(lambda x: (x - img_min) / (img_max - img_min) * (255.0 - 0.0) + 0.0)
            img = normalizer(img)

        img = img / 255.0

        img = conv2d(img, gauss_filter(self.std))
        img_mag, _ = compute_gradient(img, self.h_thd / 255.0)

        img = hessian_determinant(img_mag)
        img[img < np.max(img) * self.nm_thd] = 0
        img = nms_max(img)

        img[img > 0] = 1

        return img
# 2)
def img2coords(img):
    # flip bc imshow does y indexes 0 - 400 top to bottom.
    img = img
    row, col = img.shape
    return np.array([(j, i) for i in range(row) for j in range(col) if img[i][j] != 0.0])

def least_squares(coords):
    # ref: https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
    x, y = coords.T
    A = np.vstack([x, np.ones(len(x))]).T
    w, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return w, b

def mse(y, ypred):
    N = y.shape[0]
    return 1 / N * sum([(y_i - ypred_i) ** 2 for y_i, ypred_i in zip(y, ypred)])

class RANSAC:
    def __init__(self, t, d, s=2, p=0.5, num_lines=4):
        self.t = t
        self.d = d
        self.p = p
        self.s = s
        self.num_lines = num_lines

    def find_lines(self, img_pp):

        img_xy = img2coords(img_pp)
        total_points = img_xy.shape[0]

        result_inliers = []
        # repeat RANSAC algo for as many lines as needed
        for i in range(self.num_lines):

            # init params
            sample_count = 0
            N = 1
            inliers = np.array([[0, 0]])
            num_inliers = self.d * (img_xy.shape[0] / total_points) - 1
            while N > sample_count or num_inliers < self.d * (
                    img_xy.shape[0] / total_points):  # change bound for d based on total inliers left

                # grab s random points, apply least squares, and find inliers
                rand_points = img_xy[np.sort(np.random.randint(img_xy.shape[0], size=self.s))]
                w, b = least_squares(rand_points)
                x, y = img_xy.T
                ypred = np.array([w * x_i + b for x_i in x])
                MSE = mse(y, ypred)
                inliers = img_xy[(y - ypred) ** 2 < self.t]
                num_inliers = len(inliers)

                # update N
                e = 1 - len(inliers) / len(y)
                N = np.log(1 - self.p) / np.log(1 - (1 - e) ** self.s)
                sample_count += 1

            # add inliers
            result_inliers += [np.array(inliers)]
            img_xy = np.array([i for i in img_xy if
                               i not in inliers])  # works really well if controlling value of d based on % of total points, might run forever for high values of d

        # return selected inliers
        return result_inliers

    def forward(self, img, img_pp):
        # find inliers, then add lines, and 3x3 regions where inliers are located
        img_ransac = img.copy()
        inliers = self.find_lines(img_pp)

        for ins in inliers:
            # 3x3 area for all inliers
            for x, y in ins:
                img_ransac[y - 1:y + 2, x - 1:x + 2] = [0, 255, 0]
            # get extreme inliers
            a_x, a_y = ins[np.argmin(ins[:, 0])]
            b_x, b_y = ins[np.argmax(ins[:, 0])]
            # draw lines
            rr, cc = line(a_y, a_x, b_y, b_x)
            img_ransac[rr, cc] = [255, 0, 0]
        # img_ransac = (img_ransac * 255).astype(np.uint8)
        return img_ransac

class CannyEdgeDetection:
    def __init__(self, gaus_sigma: Union[float, int], gaus_kernel_size: int, sobel_threshold: int):
        self.gaus_sigma = gaus_sigma
        self.gaus_kernel_size = gaus_kernel_size
        self.sobel_threshold = sobel_threshold
        self._check_params()

    def non_maximum_suppression(self, img: np.array) -> np.array:
        filtery = np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ])
        filterx = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])
        orig_height = img.shape[0]
        orig_width = img.shape[1]
        # UNIT CIRCLE
        horizontal_angles = list(range(0, 30)) + list(range(150, 210)) + list(range(330, 360))
        diagonal_angle = list(range(30, 60)) + list(range(120, 150)) + list(range(210, 240)) + list(range(300, 330))
        vertical_angles = list(range(60, 120)) + list(range(240, 300))

        all_angles = horizontal_angles + diagonal_angle + vertical_angles
        assert len(all_angles) == 360
        counter = defaultdict(int)
        for i in all_angles:
            counter[i] += 1
        assert max(list(counter.values())) == 1
        shift = 3
        gray = self.convert_to_gray(img)

        direction_counter = defaultdict(int)
        nms = np.zeros(shape=(img.shape[0], img.shape[1]), dtype=np.uint8)
        for yidx in range(orig_height - shift):
            for xidx in range(orig_width - shift):
                row = gray[yidx:yidx + shift, xidx:xidx + shift]
                Gy = np.sum(row * filtery, axis=(0, 1))
                Gx = np.sum(row * filterx, axis=(0, 1))
                theta = np.arctan2(Gx, Gy) * (180 / np.pi)
                temp_theta = int(np.floor(theta))
                if temp_theta < 0:
                    temp_theta = 360 + temp_theta
                # assert False
                if temp_theta in horizontal_angles:
                    direction_counter['Horizontal Edge'] += 1
                    nb_1 = yidx - 1, xidx
                    nb_2 = yidx + 1, xidx
                elif temp_theta in diagonal_angle:
                    nb_1 = yidx + 1, xidx - 1
                    nb_2 = yidx - 1, xidx + 1
                    direction_counter['Diagonal Edge'] += 1
                elif temp_theta in vertical_angles:
                    nb_1 = yidx, xidx - 1
                    nb_2 = yidx, xidx + 1
                    direction_counter['Vertical Edge'] += 1
                else:
                    raise Exception('Angle Not Mapped!!')
                current = gray[yidx, xidx]
                nb_1 = gray[nb_1]
                nb_2 = gray[nb_2]
                if all(current > i for i in [nb_1, nb_2]):
                    nms[yidx, xidx] = 255
        print(f'Edges Found: {direction_counter}')
        return nms

    def _check_params(self) -> None:
        try:
            assert (self.gaus_sigma * 3) <= (
                    self.gaus_kernel_size / 2.0), f'ERROR: Kernel too small for sigma={self.gaus_sigma}'
        except AssertionError:
            exc = traceback.format_exc()
            accepted_gaus_kernel = self.gaus_kernel_size / 2.0
            if accepted_gaus_kernel < 3:
                message = f'[ERROR] kernel_size < 3'
            else:
                message = f'If gaus_sigma = {self.gaus_sigma}, gaus_kernel_size >= {accepted_gaus_kernel}\n' \
                          f'If gaus_kernel_size = {self.gaus_kernel_size}, gaus_sigma <= {round((self.gaus_kernel_size / 2.0) / 3.0, 2)}'
            raise Exception(f'{exc}\n{message}')

    def convert_to_gray(self, img: np.array) -> np.array:
        gray = np.dot(img[..., :3], np.array([0.299, 0.587, 0.114]))
        return gray

    def gaussian_filter(self, img: np.array) -> np.array:
        original_height = img.shape[0]
        original_width = img.shape[1]
        kernel = np.zeros((self.gaus_kernel_size, self.gaus_kernel_size, 1), np.float32)
        shift = self.gaus_kernel_size // 2
        extra_shift = 0 if self.gaus_kernel_size % 2 == 0 else 1
        for x in range(-shift, shift):
            for y in range(-shift, shift):
                x1 = 2 * np.pi * (self.gaus_sigma ** 2)
                x2 = np.e**(-(x ** 2 + y ** 2) / (2 * self.gaus_sigma ** 2))
                kernel[x + shift + extra_shift, y + shift + extra_shift] = (1 / x1) * x2

        shift = ceil((self.gaus_kernel_size-1 )/ 2)

        img_padded = np.zeros(shape = (original_height+(shift*2), original_width +(shift*2), 3), dtype = np.uint8)
        img_padded[shift:original_height + shift, shift:original_width+shift, :] = img
        img_padded = img_padded / 255.0
        padded_height, padded_width, _ = img_padded.shape
        smoothed = np.zeros_like(img_padded, dtype=np.float32)
        for yidx in range(shift, original_height + shift, 1):
            ymin = yidx - shift
            ymax = yidx + shift + extra_shift
            assert (0 <= ymin <= padded_height) and (0 <= ymax <= padded_height), f'y error  = {ymin},{ymax}'
            for xidx in range(shift, original_width + shift, 1):
                xmin = xidx - shift
                xmax = xidx + shift + extra_shift
                assert (0 <= xmin <= padded_width) and (0 <= xmax <= padded_width), f'x error = {xmin},{xmax}'
                row = img_padded[ymin:ymax, xmin:xmax] * kernel
                pix = np.sum(row, axis=(0, 1))

                smoothed[yidx, xidx] = pix
        smoothed = self._normalize(smoothed)
        smoothed_slice = smoothed[shift:original_height + shift, shift:original_width+shift, :]
        return smoothed_slice


    def _normalize(self, img: np.array) -> np.array:
        img = img * 255.0 / img.max()
        img = img.astype(np.uint8)
        return img

    def sobel_filter(self, img: np.array) -> np.array:
        filtery = np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ])
        filterx = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])

        orig_height, orig_width = img.shape[0], img.shape[1]
        sobely = np.zeros(shape=img.shape, dtype=np.float32)
        sobelx = np.zeros(shape=img.shape, dtype=np.float32)

        shift = 3
        # gray = self.convert_to_gray(img)
        gray = img.copy()
        for yidx in range(0, orig_height - shift):
            for xidx in range(0, orig_width - shift):
                row = gray[yidx:yidx + shift, xidx:xidx + shift]
                sobelx[yidx, xidx] = np.sum(row * filterx, axis=(0, 1))
                sobely[yidx, xidx] = np.sum(row * filtery, axis=(0, 1))

        sobel = np.sqrt(np.square(sobelx) + np.square(sobely))
        sobel = self._normalize(sobel)
        sobel[np.where(sobel < self.sobel_threshold)] = 0

        return sobel

    def process_canny(self, img: np.array):
        blur = self.gaussian_filter(img)
        custom_sobel = self.sobel_filter(blur)
        custom_nms = self.non_maximum_suppression(custom_sobel)
        return custom_nms

class Hough(CannyEdgeDetection):
    def __init__(self, original_img: np.array, gaus_sigma: Union[float, int], gaus_kernel_size: int, sobel_threshold: int):
        super().__init__(gaus_sigma=gaus_sigma, gaus_kernel_size=gaus_kernel_size, sobel_threshold=sobel_threshold)
        #        super().__init__(gaus_sigma=.5, gaus_kernel_size=5, sobel_threshold=40)

        self.original_img = original_img
        self.img_pp = self.process_canny(original_img)

    def _get_bounds(self, v: Union[float, int], bound: int, window_piece: Union[float, int]):

        min_v = 0 if v - window_piece < 0 else v - window_piece
        max_v = bound if (v + window_piece + 1) > bound else v + window_piece + 1
        return int(np.ceil(min_v)), int(np.ceil(max_v))
    def _draw_hough_lines(self, rho, theta) -> None:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        y1 = int(x0 + 1000 * (-b))
        x1 = int(y0 + 1000 * (a))
        y2 = int(x0 - 1000 * (-b))
        x2 = int(y0 - 1000 * (a))
        cv2.line(self.original_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    def process(self, max_peaks: int, local_maxima_window: int) -> np.array:
        img_height = self.img_pp.shape[0]
        img_width = self.img_pp.shape[1]
        diagonal = int(np.ceil(np.sqrt((img_width**2) + (img_height**2))))

        thetas = np.deg2rad([i for i in range(0, 181)])
        theta_len = len(thetas)
        rhos = np.linspace(-diagonal, diagonal, diagonal * 2)

        # hough_accumulator = np.zeros(shape=(diagonal, theta_len), dtype = np.int64)
        hough_accumulator = np.zeros(shape = (len(rhos), len(thetas)), dtype = np.int64)
        yidx_list, xidx_list = np.where(self.img_pp == 255)
        cos_list = [np.cos(theta) for theta in thetas]
        sin_list = [np.sin(theta) for theta in thetas]
        new_yidx_list = [
            np.array([int(np.round((xidx * cos_list[idx]) + (yidx * sin_list[idx]))) + diagonal for idx, theta in enumerate(thetas)])
            for xidx, yidx in zip(yidx_list, xidx_list)
        ]
        del cos_list, sin_list
        new_xidx = np.array(np.arange(0, theta_len))
        for new_yidx in new_yidx_list:
            hough_accumulator[new_yidx, new_xidx] += 1

        temp_accum = hough_accumulator.copy().ravel()
        shift = local_maxima_window // 2
        for peak in range(max_peaks):
            max_flatten_idx = np.argmax(temp_accum)
            rho, theta = np.unravel_index(max_flatten_idx, hough_accumulator.shape)
            min_theta, max_theta = self._get_bounds(theta, hough_accumulator.shape[1], shift)
            min_rho, max_rho = self._get_bounds(rho, hough_accumulator.shape[0], shift)
            for yidx in range(min_rho, max_rho):
                for xidx in range(min_theta, max_theta):
                    temp_accum[max_flatten_idx] = 0
                    if (xidx == min_theta or xidx == (max_theta - 1)):
                        hough_accumulator[yidx, xidx] = 255
                    if (yidx == min_rho or yidx == (max_rho - 1)):
                        hough_accumulator[yidx, xidx] = 255
            rho = rhos[rho]
            theta = thetas[theta]
            self._draw_hough_lines(rho, theta)
        hough_accumulator = cv2.resize(
            cv2.cvtColor(hough_accumulator.astype(np.uint8), cv2.COLOR_GRAY2BGR),
            (self.original_img.shape[1], self.original_img.shape[0]))
        stacked = [
            self.original_img, cv2.cvtColor(self.img_pp, cv2.COLOR_GRAY2BGR), hough_accumulator
        ]
        return np.hstack(stacked)

def main(filename: str, hessian_thd: float = 95, ransac_d: float = -1, hough_max_peaks: int = 5, hough_local_window: int = 3,
         gaus_kernal_size: int = 5, sobel_threshol: int = 40, gaus_sigma: Union[float, int] = .5,
         ransac_t: int = 8, ransac_p: float = .5, ransac_num_lines: int = 4) -> None:

    original_img = cv2.imread(filename)
    ransac_stack = [original_img]
    #set rgb to true if rgb image
    lpp = LinePreprocess(hessian_thd, 0.1, rgb=original_img.shape[-1]==3)
    img_pp = lpp(original_img)
    ransac_stack.append(cv2.cvtColor((img_pp * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))

    print('Running RANSAC...')
    
    ld = RANSAC(
        t = ransac_t, d = int(img_pp.shape[1] * .1) if ransac_d < 0 else ransac_d, p = ransac_p, num_lines=ransac_num_lines
    )
    ransac_img = ld.forward(original_img, img_pp)
    ransac_stack.append(cv2.cvtColor(ransac_img, cv2.COLOR_RGB2BGR))
    ransac_stack = np.hstack(ransac_stack)

    print('~' * 100)
    print('Running Hough...')
    hough = Hough(original_img, gaus_sigma=gaus_sigma, gaus_kernel_size=gaus_kernal_size, sobel_threshold=sobel_threshol)
    hough_stack = hough.process(max_peaks=hough_max_peaks, local_maxima_window=hough_local_window)
    final_stacked = np.vstack([ransac_stack, hough_stack])
    # final_stacked = np.vstack([hough_stack, hough_stack])

    cv2.imwrite(f"ld-results-{int(time.time())}.png", final_stacked)

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) > 0:
        try:
            if args[0]=="p":
                args[1], args[2], args[3], args[4] = 95, 8, -1, 3
            else:
                args[1] = float(args[1])
                args[2] = float(args[2])
                args[3] = float(args[3])
                args[4] = int(args[4])

        except:
          raise Exception("Usage: [python/python3] LineDetection.py [p/np: default/custom] (float: hessian threshold) (float: distance threshold) (float: required inliers) (int: dim. of bin accum.) (str: image path)") 
        h_thd, t, d, dim_accum, fp = args[1:]
    
    else:
        #if d < 0, use 10% of points in image as number of inliers
        h_thd, t, d, dim_accum, fp = 95, 8, -1, 3, 'road.png' 
        
    print(f"Running line detection with:\nPreprocess:\n hessian threshold: {h_thd}\nRANSAC:\n distance threshold: {t}\n required inliers: {'10% of preprocess points' if d < 0 else d}\nHough:\n dimension of bins of accum.: {dim_accum}\nFile: {fp}")

    main(
        filename=fp,
        hessian_thd=h_thd,
        ransac_t=t, ransac_d=d, ransac_num_lines=4, ransac_p=.5,
        hough_max_peaks=4, hough_local_window=dim_accum,
        gaus_kernal_size=7, sobel_threshol=100, gaus_sigma=1
    )
