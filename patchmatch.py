import ctypes
import os.path as pth
import numpy as np

PMLIB = ctypes.CDLL(pth.join(pth.dirname(__file__), 'bin', 'ebsynth'))

BackendAvailable = PMLIB.ebsynthBackendAvailable
BackendAvailable.argtypes = [ctypes.c_int]
BackendAvailable.restype = ctypes.c_int
ebsynthRun = PMLIB.ebsynthRun
ebsynthRun.argtypes = [ctypes.c_int,
                       ctypes.c_int,
                       ctypes.c_int,
                       ctypes.c_int,
                       ctypes.c_int,
                       ctypes.c_void_p,
                       ctypes.c_void_p,
                       ctypes.c_int,
                       ctypes.c_int,
                       ctypes.c_void_p,
                       ctypes.c_void_p,
                       ctypes.POINTER(ctypes.c_float),
                       ctypes.POINTER(ctypes.c_float),
                       ctypes.c_float,
                       ctypes.c_int,
                       ctypes.c_int,
                       ctypes.c_int,
                       ctypes.POINTER(ctypes.c_int),
                       ctypes.POINTER(ctypes.c_int),
                       ctypes.POINTER(ctypes.c_int),
                       ctypes.c_int,
                       ctypes.c_void_p,
                       ctypes.c_void_p]
ebsynthRun.restype = None


def pyramidLevelSize(width: int, height: int, level: int) -> tuple:
    width *= np.power(2.0, -level)
    height *= np.power(2.0, -level)
    return (np.around(width), np.around(height))


def inpaint(image: np.ndarray, mask: np.ndarray, weight=1, uniformity=3500, patchsize=5, pyramidlevels=-1, searchvoteiters=6, patchmatchiters=4, stopthreshold=1, extrapass3x3=0) -> np.ndarray:
    """
    PatchMatch based inpainting proposed in:

        PatchMatch : A Randomized Correspondence Algorithm for Structural Image Editing
                C.Barnes, E.Shechtman, A.Finkelstein and Dan B.Goldman
                SIGGRAPH 2009

    Implementation modified from:

        https://github.com/jamriska/ebsynth

    Args:
        image ([h*w*c] np.ndarray): the input image, can be either greyscale, RGB or stacked SVBRDF
        mask ([h*w*1] np.ndarray): the mask of the hole to be filled

    Returns:
        output ([h*w*c] np.ndarray): the inpainted image, of the same size as the input image
    """
    BACKEND_CUDA = 0x0002
    EBSYNTH_VOTEMODE = 0x0002
    if BackendAvailable(ctypes.c_int(BACKEND_CUDA)) != 1:
        raise NotImplementedError('CUDA is required to execute the PatchMatch algorithm')

    image_height, image_width, image_channel = image.shape
    mask_height, mask_width, mask_channel = mask.shape

    image_data = image.flatten()
    mask_data = mask.flatten()
    inv_mask_data = np.invert(mask_data)

    image_weights = [1.0 / image_channel for _ in range(image_channel)]
    image_weights_array = (ctypes.c_float*len(image_weights))(*image_weights)
    mask_weights = [weight / mask_channel for _ in range(mask_channel)]
    mask_weights_array = (ctypes.c_float * len(mask_weights))(*mask_weights)

    max_pyramid_levels = 0
    for level in range(32, -1, -1):
        if min(pyramidLevelSize(image_width, image_height, level)) >= 2 * patchsize + 1:
            max_pyramid_levels = level + 1
            break
    if pyramidlevels == -1:
        pyramidlevels = max_pyramid_levels
    pyramidlevels = min(pyramidlevels, max_pyramid_levels)

    searchVoteItersPerLevel = [searchvoteiters for _ in range(pyramidlevels)]
    search_vote_iters_array = (ctypes.c_int * len(searchVoteItersPerLevel))(*searchVoteItersPerLevel)
    patchMatchItersPerLevel = [patchmatchiters for _ in range(pyramidlevels)]
    patch_match_iters_array = (ctypes.c_int * len(patchMatchItersPerLevel))(*patchMatchItersPerLevel)
    stopThresholdPerLevel = [stopthreshold for _ in range(pyramidlevels)]
    stop_threshold_array = (ctypes.c_int*len(stopThresholdPerLevel))(*stopThresholdPerLevel)

    print("uniformity:", uniformity)
    print("weight:", weight)
    print("patchsize", patchsize)
    print("pyramidlevels:", pyramidlevels)
    print("searchvoteiters:", searchvoteiters)
    print("patchmatchiters:", patchmatchiters)
    print("stopthreshold:", stopthreshold)
    print("extrapass3x3: {}".format('yes' if extrapass3x3 != 0 else 'no'))
    print("backend: cuda")

    output_data = np.empty_like(image_data)

    ebsynthRun(ctypes.c_int(BACKEND_CUDA),
               ctypes.c_int(image_channel),
               ctypes.c_int(mask_channel),
               ctypes.c_int(image_width),
               ctypes.c_int(image_height),
               image_data.ctypes.data_as(ctypes.c_void_p),
               mask_data.ctypes.data_as(ctypes.c_void_p),
               ctypes.c_int(mask_width),
               ctypes.c_int(mask_height),
               inv_mask_data.ctypes.data_as(ctypes.c_void_p),
               None,
               image_weights_array,
               mask_weights_array,
               ctypes.c_float(uniformity),
               ctypes.c_int(patchsize),
               ctypes.c_int(EBSYNTH_VOTEMODE),
               ctypes.c_int(pyramidlevels),
               search_vote_iters_array,
               patch_match_iters_array,
               stop_threshold_array,
               ctypes.c_int(extrapass3x3),
               None,
               output_data.ctypes.data_as(ctypes.c_void_p)
               )

    return output_data.reshape((image_height, image_width, image_channel))
