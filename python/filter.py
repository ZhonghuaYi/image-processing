"""
    Functions here are about image filtering.
"""
import numpy as np
from Spatial import grayscale_mapping, grayscale_intercept

import exception


def convolution(in_pic, kernel):
    """
    对图像进行卷积，kernel的shape需要是奇数。
    :param in_pic: 输入图像
    :param kernel: 卷积核
    :return: 卷积结果(numpy.float32)
    """
    in_pic_shape = in_pic.shape
    out = np.zeros((in_pic_shape[0] + kernel.shape[0] - 1, in_pic_shape[1] + kernel.shape[1] - 1), dtype=np.float32)
    exception.check_kernel_shape(kernel.shape)

    index = kernel.shape[0] // 2, kernel.shape[1] // 2
    out[index[0]:index[0] + in_pic_shape[0], index[1]:index[1] + in_pic_shape[1]] = in_pic
    out_copy = out.copy()

    for i in range(in_pic_shape[0]):
        for j in range(in_pic_shape[1]):
            convolution_sum = 0
            convolution_sum += np.sum(out_copy[i: i + kernel.shape[0], j:j + kernel.shape[1]] * kernel)
            out[i][j] = round(convolution_sum)

    return out[index[0]:index[0] + in_pic_shape[0], index[1]:index[1] + in_pic_shape[1]]


# generate gaussian kernel, it's always a square.
def gauss_kernel(kernel_size, sigma):
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    if sigma <= 0:
        sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8

    s = sigma ** 2
    sum_val = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - center, j - center

            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / 2 * s)
            sum_val += kernel[i, j]

    kernel = kernel / sum_val

    return kernel


def box_blur(in_pic, kernel_shape):
    kernel = np.ones((kernel_shape[0], kernel_shape[1])) / (kernel_shape[0] * kernel_shape[1])
    out = convolution(in_pic, kernel)
    return out.astype(np.uint8)


def median_blur(in_pic, kernel_shape):
    exception.check_kernel_shape(kernel_shape)
    index = kernel_shape[0]//2, kernel_shape[1]//2
    media_index = round(kernel_shape[0] * kernel_shape[1] / 2)
    out = in_pic.copy()
    in_pic_extend = np.zeros((in_pic.shape[0] + kernel_shape[0] - 1, in_pic.shape[1] + kernel_shape[1] - 1),
                             dtype=in_pic.dtype)
    in_pic_extend[index[0]:index[0] + in_pic.shape[0], index[1]:index[1] + in_pic.shape[1]] = in_pic
    for i in range(in_pic.shape[0]):
        for j in range(in_pic.shape[1]):
            sort = np.sort(in_pic_extend[i:i+kernel_shape[0], j:j+kernel_shape[1]].flatten())
            out[i][j] = np.min(sort[media_index])

    return out
