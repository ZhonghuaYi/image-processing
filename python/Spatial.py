"""
    Functions from this file are about image processing in spatial domain.
    Only apply on 8-bit grayscale.
"""

import numpy as np
import math


# shift the image. Option decide the way out image is cut.
def shift(in_pic, pic_shift, option=0, out_shape=(0, 0), cut=(0, 0)):
    in_shape = in_pic.shape
    x_shift, y_shift = pic_shift[0], pic_shift[1]
    shape = (in_shape[0] + abs(x_shift), in_shape[1] + abs(y_shift))
    full_out = np.zeros(shape=shape, dtype=np.uint8)
    pic_origin = [0, 0]
    cut_origin = [0, 0]
    out_x_size, out_y_size = out_shape
    if out_shape[0] == 0:
        out_x_size = in_shape[0]
    if out_shape[1] == 0:
        out_y_size = in_shape[1]

    if x_shift < 0:
        cut_origin[0] = abs(x_shift)
    else:
        pic_origin[0] = abs(x_shift)
    if y_shift < 0:
        cut_origin[1] = abs(y_shift)
    else:
        pic_origin[1] = abs(y_shift)

    full_out[pic_origin[0]:pic_origin[0] + in_shape[0], pic_origin[1]:pic_origin[1] + in_shape[1]] = in_pic

    if option == 1:
        out = full_out[cut_origin[0]:cut_origin[0] + out_x_size, cut_origin[1]:cut_origin[1] + out_y_size]

    elif option == 2:
        cut_origin = cut
        out = full_out[cut_origin[0]:, cut_origin[1]:]

    else:
        out = full_out

    return out


# angle is by anticlockwise.
def rotate(in_pic, angle):
    pass


def rgb2gray(in_pic):
    if in_pic.ndim == 2:
        out = in_pic.copy()

    else:
        out = in_pic[:, :, 0] * 0.114 + in_pic[:, :, 1] * 0.587 + in_pic[:, :, 2] * 0.229
        out = out.astype(np.uint8)

    return out


def resize(in_pic, resize_shape, method='nearest'):
    out = np.zeros(shape=resize_shape, dtype=np.uint8)
    in_size = in_pic.shape
    x_distance = float(in_size[0]) / resize_shape[0]
    y_distance = float(in_size[1]) / resize_shape[1]
    if method == 'bilinear':
        for i in range(resize_shape[0]):
            for j in range(resize_shape[1]):
                x, y = i * x_distance, j * y_distance
                x1, x2 = math.floor(x), math.ceil(x)
                y1, y2 = math.floor(y), math.ceil(y)
                f_1y = (float(in_pic[x1][y2]) - in_pic[x1][y1]) * (y - y1) + in_pic[x1][y1]
                f_2y = (float(in_pic[x2][y2]) - in_pic[x2][y1]) * (y - y1) + in_pic[x2][y1]
                f_xy = (f_2y - f_1y) * (x - x1) + f_1y
                out[i][j] = round(f_xy)

    else:
        for i in range(resize_shape[0]):
            for j in range(resize_shape[1]):
                x, y = round(i * x_distance), round(j * y_distance)
                out[i][j] = in_pic[x][y]

    return out


# map [in_low, in_high] into [out_low, out_high] linearly.
def grayscale_mapping(in_pic, in_low, in_high, out_low, out_high):
    if in_low == 0:
        slope_1 = 0

    else:
        slope_1 = out_low / in_low

    slope_2 = (out_high - out_low) / (in_high - in_low)
    if in_high == 255:
        slope_3 = 0

    else:
        slope_3 = (255 - out_high) / (255 - in_high)

    pic_shape = in_pic.shape
    out = np.zeros(pic_shape, dtype=np.float64)
    for i in range(pic_shape[0]):
        for j in range(pic_shape[1]):
            gray_in = in_pic[i][j]
            gray_out = 0

            if (gray_in < in_low) and (gray_in > 0):
                gray_out = round(slope_1 * gray_in)

            elif (gray_in < in_high) and (gray_in > in_low):
                gray_out = round(out_low + slope_2 * (gray_in - in_low))

            elif (gray_in < 255) and (gray_in > in_high):
                gray_out = round(out_high + slope_3 * (gray_in - in_high))

            out[i][j] = gray_out

    return out


# negative value will be 0, and value bigger than 255 will just be 255.
def grayscale_intercept(in_pic):
    out = in_pic.copy()
    for i in np.nditer(out, op_flags=['readwrite']):
        if i < 0:
            i[...] = 0
        if i > 255:
            i[...] = 255

    return out


# in_pic is in gray domain.
def get_histogram(in_pic, scale=256):
    histogram = np.zeros(scale)
    pic_size = in_pic.size
    for i in in_pic.flat:
        histogram[i] += 1

    histogram /= pic_size
    return histogram


def plot_histogram(histogram, title="Histogram", xlabel="Grayscale", ylabel="Probability"):
    from matplotlib import pyplot as plt
    x = np.arange(histogram.size)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, histogram)
    plt.show()


# get the negative of the in_pic.
def spatial_reverse(in_pic):
    out = 255 - in_pic
    return out


# get the log transform and out is in [0, 255]
def log_transform(in_pic):
    from math import log
    mat_max = np.max(in_pic)
    c = 255 / log(1 + mat_max)
    # use 1e-5 to avoid warning.
    out = c * np.log(1 + in_pic + 1e-5)
    out = out.astype(np.uint8)
    return out


def gamma_transform(in_pic, gamma):
    gamma = float(gamma)
    c = 255 / pow(255, gamma)
    out = c * np.power(in_pic, gamma)
    out = out.astype(np.uint8)
    return out


# map [in_low, in_high] into [out_low, out_high], but out's dtype is uint8.
def contrast_stretching(in_pic, in_low, in_high, out_low, out_high):
    out = grayscale_mapping(in_pic, in_low, in_high, out_low, out_high)
    out = out.astype(np.uint8)
    return out


# map [in_low, in_high} into an constant gray value.
def gray_level_slicing(in_pic, in_low, in_high, constant):
    out = in_pic.copy()
    for i in np.nditer(out, op_flags=['readwrite']):
        if (i > in_low) and (i < in_high):
            i[...] = constant

    return out


# get one bit plane from 0~7 plane.
def bit_plane_slicing(in_pic, layer):
    out = in_pic.copy()
    for i in np.nditer(out, op_flags=['readwrite']):
        # bin_value is 8-bit width.
        bin_value = bin(i)[2:].zfill(8)
        layer_bit = bin_value[-(int(layer) + 1)]
        i[...] = int(layer_bit) * 255

    return out


# get the cumulative distribution function(CDF) of in_pic.
def cdf(in_pic_histogram):
    scale = in_pic_histogram.size
    transform = np.zeros(scale, dtype=np.uint8)
    temp = 0
    for i in range(scale):
        temp = temp + in_pic_histogram[i]
        transform[i] = (scale - 1) * temp

    return transform


def histogram_equalize(in_pic):
    out = in_pic.copy()
    in_pic_histogram = get_histogram(in_pic)
    transform = cdf(in_pic_histogram)
    for i in np.nditer(out, op_flags=['readwrite']):
        i[...] = transform[i]

    return out


# variable match is a given histogram for histogram matching.
def histogram_matching(in_pic, match):
    out = in_pic.copy()
    in_pic_histogram = get_histogram(in_pic)
    in_transform = cdf(in_pic_histogram)
    match_transform = cdf(match)
    match_transform_inverse = np.zeros(match_transform.size, dtype=np.uint8)
    for i in range(match_transform.size):
        j = match_transform[-i - 1]
        match_transform_inverse[j] = 255 - i

    for i in range(match_transform_inverse.size):
        if i == 0:
            pass

        else:
            if match_transform_inverse[i] == 0:
                match_transform_inverse[i] = match_transform_inverse[i - 1]

    for i in np.nditer(out, op_flags=['readwrite']):
        i[...] = in_transform[i]
        i[...] = match_transform_inverse[i]

    return out


def match_histogram(in_pic, match):
    out = in_pic.copy()
    in_histogram = get_histogram(in_pic)
    in_cdf = cdf(in_histogram)
    match_cdf = cdf(match)
    # 构建累积概率误差矩阵
    diff_cdf = np.zeros((256, 256))
    for k in range(256):
        for j in range(256):
            diff_cdf[k][j] = np.abs(int(in_cdf[k]) - int(match_cdf[j]))

    # 生成映射表
    lut = np.argmin(diff_cdf, axis=1)
    # lut = np.zeros((256, ), dtype=np.uint8)
    # for m in range(256):
    #     min_val = diff_cdf[m][0]
    #     index = 0
    #     for n in range(256):
    #         if min_val > diff_cdf[m][n]:
    #             min_val = diff_cdf[m][n]
    #             index = n
    #     lut[m] = index
    
    for i in np.nditer(out, op_flags=['readwrite']):
        i[...] = lut[i]

    return out


def remove_black_border(in_pic):
    out = in_pic.copy()
    while np.all(out[0, :] == 0):
        out = np.delete(out, 0, axis=0)

    while np.all(out[-1, :] == 0):
        out = np.delete(out, -1, axis=0)

    while np.all(out[:, 0] == 0):
        out = np.delete(out, 0, axis=1)

    while np.all(out[:, -1] == 0):
        out = np.delete(out, -1, axis=1)

    return out

