"""
    Define custom exceptions.
"""


def check_kernel_shape(kernel_shape):
    try:
        if kernel_shape[0] % 2 == 0 or kernel_shape[1] % 2 == 0:
            raise Exception("kernel shape can't be even.")
    except Exception as e:
        print(e)
        exit(1)
