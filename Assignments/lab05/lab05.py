# -*- coding: utf-8 -*-
"""Lab 05 — Harris Corner Detector.

Detect corner points in an image from scratch — no cv2.cornerHarris. The
work is split into small steps, each one a method below that takes an array
in and returns the result out:

  1. gaussian_kernel()                    a 2D Gaussian weighting window
  2. compute_gradients(image)             Sobel gradients (Ix, Iy)
  3. gradient_products(Ix, Iy)            Ix^2, Iy^2, Ix*Iy
  4. smooth_products(Ixx, Iyy, Ixy)       Gaussian-weighted sums (A, B, C)
  5. det_and_trace(A, B, C)               Det(M) and Trace(M) at every pixel
  6. harris_response(det, trace)          Harris response R per pixel
  7. threshold_response(response)         weak responses set to zero
  8. find_peaks(thresh_response)          one (row, col) per corner
  9. det_coords(image)                    put the steps together

Run evaluate_lab05.py after each step to see your progress.
"""
import numpy as np
from scipy import signal
import os
import sys
from skimage.feature import peak_local_max

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from helpers.dataloader import DATA_DIR, download_image, norm_img, rescale_crop  # noqa: F401

CHECKERBOARD_IMAGE_URL = (
    "https://github.com/tonyscan6003/etivities/blob/main/"
    "10x10_checkered_board.png?raw=true"
)

class HarrisCorner:
    """Harris Corner Detector, built from the second-moment matrix of the
    image gradients.

    All settings have defaults, so no arguments are needed:

        hc     = HarrisCorner()
        coords = hc.det_coords(image)

    Any setting can be changed with a keyword argument, e.g.:

        hc = HarrisCorner(sigma=3, min_distance=10)

    Use these settings (self.sigma, self.alpha, ...) inside your methods
    rather than typing the numbers in directly.
    """

    def __init__(self, **kwargs):
        self.sigma         = kwargs.get('sigma',         2)
        self.alpha         = kwargs.get('alpha',         0.05)
        self.threshold_val = kwargs.get('threshold_val', 0.001)
        self.min_distance  = kwargs.get('min_distance',  5)

        # Intermediate results, filled in by det_coords() (step 9)
        self.grad_x          = None
        self.grad_y          = None
        self.grad_products   = None
        self.smoothed        = None
        self.det             = None
        self.trace           = None
        self.corner_response = None
        self.thresh_response = None
        self.coords          = None

    def gaussian_kernel(self):
        k = int(6 * self.sigma + 1)
        half = (k - 1) // 2
        x = np.arange(-half, half + 1)
        g = np.exp(-x ** 2 / (2 * self.sigma ** 2)) / (self.sigma * np.sqrt(2 * np.pi))
        return np.outer(g, g)

    def compute_gradients(self, image):
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_y = np.array([[ 1, 2, 1], [ 0, 0, 0], [-1, -2, -1]])
        Ix = signal.convolve2d(image, kernel_x, boundary='symm', mode='same')
        Iy = signal.convolve2d(image, kernel_y, boundary='symm', mode='same')
        return Ix, Iy

    def gradient_products(self, Ix, Iy):
        return Ix * Ix, Iy * Iy, Ix * Iy

    def smooth_products(self, Ixx, Iyy, Ixy):
        window = self.gaussian_kernel()
        A = signal.convolve2d(Ixx, window, boundary='symm', mode='same')
        B = signal.convolve2d(Iyy, window, boundary='symm', mode='same')
        C = signal.convolve2d(Ixy, window, boundary='symm', mode='same')
        return A, B, C

    def det_and_trace(self, A, B, C):
        det = A * B - C ** 2
        trace = A + B
        return det, trace

    def harris_response(self, det, trace):
        return det - self.alpha * trace ** 2

    def threshold_response(self, response):
        return np.where(response > self.threshold_val, response, 0)

    def find_peaks(self, thresh_response):
        return peak_local_max(thresh_response, min_distance=self.min_distance)

    def det_coords(self, image):
        self.grad_x, self.grad_y = self.compute_gradients(image)
        self.grad_products = self.gradient_products(self.grad_x, self.grad_y)
        self.smoothed = self.smooth_products(*self.grad_products)
        self.det, self.trace = self.det_and_trace(*self.smoothed)
        self.corner_response = self.harris_response(self.det, self.trace)
        self.thresh_response = self.threshold_response(self.corner_response)
        self.coords = self.find_peaks(self.thresh_response)
        return self.coords
