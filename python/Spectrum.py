"""
    Functions in this file are about image processing in spectrum.
"""
import numpy as np
import Spatial


# only use on gray image.
def spectrum_draw(in_pic):
    f = np.fft.fft2(in_pic)
    f_shift = np.fft.fftshift(f)
    f_amplitude_log = Spatial.log_transform(np.abs(f_shift))
    from matplotlib import pyplot as plt
    plt.subplot(121)
    plt.imshow(in_pic, 'gray')
    plt.title('origin')
    plt.subplot(122)
    plt.imshow(f_amplitude_log, 'gray')
    plt.title('fourier')
    plt.show()
