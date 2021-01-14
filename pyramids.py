import numpy as np
from scipy.misc import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve
import os


MAX_PIX_LEVEL = 255
GRAYSCALE = 1
COLOR = 2
DIM_BORDER = 32
RGB_CHANNELS = 3


def relpath(filename):
    """
    gets files from external path
    :param filename: file to be opened
    """
    return os.path.join(os.path.dirname(__file__), filename)


def read_image(filename, representation):
    """
    opens image as matrix
    :param filename: name of image
    :param representation: 1 if grayscale, 2 if RGB
    :return: np.array float64 of given image
    """
    image = imread(filename)
    image_float = image.astype(np.float64) / MAX_PIX_LEVEL
    if representation == GRAYSCALE:
        image_float_gray = rgb2gray(image_float)
        return image_float_gray
    return image_float


def create_filter(size):
    """
    helper function to create gaussian filter vector of given size
    :param size: size of wanted filter
    :return: the wanted filter, normalized
    """
    vec_filter = np.array([1])
    a = np.array([1, 1])
    convo_multiplier = np.convolve(a,a)
    counter = 1
    while counter < size:
        vec_filter = np.convolve(vec_filter, convo_multiplier)
        counter += 2
    filter_sum = np.sum(vec_filter)
    normalized_filter = vec_filter / filter_sum
    return normalized_filter.reshape(1,normalized_filter.size)


def reduce(im, filter_vec):
    """
    helper function, reduces given image by 2
    :param im: image before reduction
    :param filter_vec: blurring filter to be used as part of reduction system
    :return: reduced image
    """
    blurred = convolve(im, filter_vec.T)
    reduced = blurred[::2,:]
    blurred = convolve(reduced, filter_vec)
    reduced = blurred[:,::2]
    return reduced


def expand(im, filter_vec):
    """
    helper function, expands given image by 2
    :param im: image before expansion
    :param filter_vec: blurring filter to be used as part of expansion system
    :return: expanded image
    """
    filter_exp = filter_vec * 2
    half_shape = (im.shape[0],im.shape[1]*2)
    full_shape = (im.shape[0]*2,im.shape[1]*2)

    half_expanded = np.zeros(half_shape)
    half_expanded[:,::2] = im
    blurred_half = convolve(half_expanded, filter_exp)

    expanded = np.zeros(full_shape)
    expanded[::2,:] = blurred_half
    blurred_exp = convolve(expanded, filter_exp.T)
    return blurred_exp


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    builds gaussian pyramid
    :param im: original image
    :param max_levels: max levels of pyramid
    :param filter_size: size of blurring filter to be used for each level
    :return: gaussian pyramid and normalized filter
    """
    filter_vec = create_filter(filter_size)
    pyr = [im]
    reduced_im = im
    for i in range(1, max_levels):
        reduced_im = reduce(reduced_im, filter_vec)
        pyr.append(reduced_im)
        y_shape, x_shape = pyr[-1].shape
        if x_shape < DIM_BORDER or y_shape < DIM_BORDER:
            break

    return pyr, filter_vec


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    builds laplacian pyramid
    :param im: original image
    :param max_levels: max levels of pyramid
    :param filter_size: size of blurring filter to be used for each level
    :return: laplacian pyramid and normalized filter
    """
    gaus_pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    if max_levels == 1:
        return gaus_pyr, filter_vec
    pyr = [im - expand(gaus_pyr[1], filter_vec)]
    for i in range(1, len(gaus_pyr) - 1):
        expanded_im = expand(gaus_pyr[i+1], filter_vec)
        pyr.append(gaus_pyr[i] - expanded_im)
    pyr.append(gaus_pyr[len(gaus_pyr) - 1])
    return pyr, filter_vec


def laplacian_to_image(lypr, filter_vec, coeff):
    """
    reconstruct image from laplacian pyramid
    :param lypr: laplacian pyramid
    :param filter_vec: blur filter
    :param coeff: list of coefficients for each level of the pyramid, used to control how much of
    each level is wanted.
    :return: reconstructed image
    """
    pyr = lypr[::-1]
    coeff = coeff[::-1]
    output_im = pyr[0]*coeff[0]
    for i in range(1, len(pyr)):
        output_im = expand(output_im, filter_vec) + pyr[i]*coeff[i]
    return output_im


def stretch(image):
    """
    helper function, stretches image values to be in range [0,1]
    :param image:
    :return:
    """
    return (image - np.min(image)) / (np.max(image) - np.min(image))


def render_pyramid(pyr, levels):
    """
    creates platform to be able to show all levels of given pyramid together and in declining order
    :param pyr: given pyramid, either gaussian or laplacian
    :param levels: levels in pyramid
    :return: stack of levels ready for display
    """
    output = stretch(pyr[0])
    y_shape = pyr[0].shape[0]
    for i in range(1, levels):
        next_im_to_add = np.zeros((y_shape, pyr[i].shape[1]))
        next_im_to_add[:pyr[i].shape[0],:pyr[i].shape[1]] = stretch(pyr[i])
        output = np.hstack((output, next_im_to_add))
    return output


def display_pyramid(pyr, levels):
    """
    displays the stack of all pyramid levels which was created in "render_pyramid" function
    :param pyr: pyramid to display, either gaussian or laplacian
    :param levels: amount of pyramid's levels.
    """
    res = render_pyramid(pyr, levels)
    plt.imshow(res, cmap='gray')
    plt.show()
    return



def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    blends to given images together
    :param im1: first image
    :param im2: second image
    :param mask: area to be switched and blended
    :param max_levels: max levels of pyramid
    :param filter_size_im: size of blur filter for the images
    :param filter_size_mask: size of blur filter for the mask
    :return: blended image
    """
    L1, filter1 = build_laplacian_pyramid(im1,max_levels,filter_size_im)
    L2, filter2 = build_laplacian_pyramid(im2,max_levels,filter_size_im)
    Gm, filter_g = build_gaussian_pyramid(mask.astype(np.float64),max_levels,filter_size_mask)
    Lout = []
    for k in range(len(L1)):
        Lout.append((Gm[k] * L1[k]) + ((1 - Gm[k]) * L2[k]))
    return laplacian_to_image(Lout,filter1,[1]*len(Lout)).clip(0,1)


def blend_RGB(im1, im2, mask, max_levels, filter_lap, filter_mask):
    """
    helper function to blend each channel of RGB in images separately
    :param im1: first image
    :param im2: second image
    :param mask: mask
    :param max_levels: max levels of pyramid
    :param filter_lap: filter for laplacian pyramid to be used for images
    :param filter_mask: filter for gaussian pyramid to be used for mask
    :return: blended image, combined all RGB channels together
    """
    output = np.zeros(im1.shape)
    for i in range(RGB_CHANNELS):
        output[:,:,i]=pyramid_blending(im1[:,:,i],im2[:,:,i],mask,max_levels,filter_lap,filter_mask)
    return output


def showSubPlots(im1, im2, mask, im_blend):
    """
    helper function, displays together to user all four params
    :param im1: first origignal image
    :param im2: second original image
    :param mask: mask
    :param im_blend: blended image
    """
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(im1)
    ax[0, 1].imshow(im2)
    ax[1, 0].imshow(mask, cmap='gray')
    ax[1, 1].imshow(im_blend)
    plt.show()


def blending_example1():
    """
    first example of my original blend, eye and earth
    :return: first image, second image, mask, blended image
    """
    im1 = read_image(relpath('externals/eye.jpg'), COLOR)
    im2 = read_image(relpath('externals/earth.jpg'), COLOR)
    mask = read_image(relpath('externals/maskEarthEye.jpg'), GRAYSCALE).astype(np.bool)
    im_blend = blend_RGB(im1, im2, mask, 4, 5, 5)
    showSubPlots(im1, im2, mask, im_blend)
    return im1, im2, mask, im_blend


def blending_example2():
    """
    second example of my original blend, trump and umbridge
    :return: first image, second image, mask, blended image
    """
    im1 = read_image(relpath('externals/umbridge.jpg'), COLOR)
    im2 = read_image(relpath('externals/trump.jpg'), COLOR)
    mask = read_image(relpath('externals/maskFace.jpg'), GRAYSCALE).astype(np.bool)
    im_blend = blend_RGB(im1, im2, mask, 5, 5, 5)
    showSubPlots(im1, im2, mask, im_blend)
    return im1, im2, mask, im_blend

