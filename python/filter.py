"""
    Functions here are about image filtering.
"""
import numpy as np
from Spatial import grayscale_mapping, grayscale_intercept


# default padding of edge is 0.
# when operate_type is 'linear', it means convolution.
# when operate_type is 'order_statistic', just require kernel's shape.
def image_kernel_operation(in_pic, operate_type='linear', kernel=None):
    in_pic_shape = in_pic.shape
    out = np.zeros((in_pic_shape[0] + kernel.shape[0] - 1, in_pic_shape[1] + kernel.shape[1] - 1))
    index = kernel.shape[0] // 2, kernel.shape[1] // 2
    out[index[0]:index[0] + in_pic_shape[0], index[1]:index[1] + in_pic_shape[1]] = in_pic
    out_copy = out.copy()
    if operate_type == 'order_statistic':
        for i in range(index[0], in_pic_shape[0] + index[0]):
            for j in range(index[1], in_pic_shape[1] + index[1]):
                temp = out_copy[i - index[0]:i + index[0], j - index[1]:j + index[1]]
                temp_median = np.median(temp)
                out[i][j] = temp_median.astype(np.uint8)

    else:
        for i in range(index[0], in_pic_shape[0] + index[0]):
            for j in range(index[1], in_pic_shape[1] + index[1]):
                convolution_sum = 0

                for s in range(-index[0], index[0] + 1):
                    for t in range(-index[1], index[1] + 1):
                        convolution_sum += out_copy[i - s][j - t] * kernel[index[0] + s][index[1] + t]

                out[i][j] = round(convolution_sum)

    temp_max = max(np.max(out), abs(np.min(out)))
    if np.min(out) < 0:
        out = grayscale_mapping(out, -temp_max, 0, -255, 0)

    if np.max(out) > 255:
        out = grayscale_mapping(out, 0, temp_max, 0, 255)

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


# allow box filtering and gaussian filtering.
# default box filtering.
def smooth(in_pic, kernel_shape, kernel_type='box', sigma=1):
    if kernel_type == 'gaussian':
        try:
            if kernel_shape[0] != kernel_shape[1]:
                raise Exception("given kernel shape is not a square.")

        except Exception as e:
            print(e)
            return None

        else:
            kernel = gauss_kernel(kernel_shape[0], sigma)

        out = image_kernel_operation(in_pic, operate_type='linear', kernel=kernel)

    elif kernel_type == 'order_statistic':
        kernel = np.zeros(shape=kernel_shape)
        out = image_kernel_operation(in_pic, operate_type='order_statistic', kernel=kernel)

    else:
        kernel = np.ones(shape=kernel_shape) / (kernel_shape[0] * kernel_shape[1])
        out = image_kernel_operation(in_pic, operate_type='linear', kernel=kernel)

    out = out.astype(np.uint8)
    return out


def sharpen(in_pic, method='laplacian', blur_method='box', kernel_shape=(3, 3), sigma=1):
    if method == 'blur':
        if blur_method == 'gaussian':
            sharpen_model = in_pic.astype(np.float64) - smooth(in_pic, kernel_shape, kernel_type='gaussian',
                                                               sigma=sigma)

        elif blur_method == 'order_statistic':
            sharpen_model = in_pic.astype(np.float64) - smooth(in_pic, kernel_shape,
                                                               kernel_type='order_statistic')

        else:
            sharpen_model = in_pic.astype(np.float64) - smooth(in_pic, kernel_shape, kernel_type='box')

    elif method == 'gradiant':
        # use sobel operator.
        sobel_x = np.asarray([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        sobel_y = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sharpen_model_x = image_kernel_operation(in_pic, operate_type='linear', kernel=sobel_x)
        sharpen_model_y = image_kernel_operation(in_pic, operate_type='linear', kernel=sobel_y)
        sharpen_model = sharpen_model_x + sharpen_model_y

    else:
        # use laplace operator.
        laplacian = np.asarray([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        sharpen_model = image_kernel_operation(in_pic, operate_type='linear', kernel=laplacian)

    temp = sharpen_model + in_pic
    out = grayscale_intercept(temp).astype(np.uint8)
    return out


# method gives the way that process negative value of convolution image
def conv_image(in_pic, kernel_type='laplacian', method='scale'):
    if kernel_type == 'gradiant':
        sobel_x = np.asarray([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        conv_result_x = image_kernel_operation(in_pic, operate_type='linear', kernel=sobel_x)
        sobel_y = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        conv_result_y = image_kernel_operation(in_pic, operate_type='linear', kernel=sobel_y)
        conv_result = conv_result_x + conv_result_y

    else:
        laplacian = np.asarray([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        conv_result = image_kernel_operation(in_pic, operate_type='linear', kernel=laplacian)

    if np.min(conv_result) < 0:
        if method == 'intercept':
            for i in np.nditer(conv_result, op_flags=['readwrite']):
                if i < 0:
                    i[...] = 0

        else:
            conv_result = conv_result - np.min(conv_result)

    if np.max(conv_result) > 255:
        conv_result = grayscale_mapping(conv_result, 0, np.max(conv_result), 0, 255)

    conv_result = conv_result.astype(np.uint8)
    return conv_result
