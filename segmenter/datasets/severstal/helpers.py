import numpy as np


def rle_to_mask(lre, shape=(1600, 256)):
    '''
    params:  rle   - run-length encoding string (pairs of start & length of encoding)
             shape - (width,height) of numpy array to return 
    
    returns: numpy array with dimensions of shape parameter
    '''
    # the incoming string is space-delimited
    runs = np.asarray([int(run) for run in lre.split(' ')])

    # we do the same operation with the even and uneven elements, but this time with addition
    runs[1::2] += runs[0::2]
    # pixel numbers start at 1, indexes start at 0
    runs -= 1

    # extract the starting and ending indeces at even and uneven intervals, respectively
    run_starts, run_ends = runs[0::2], runs[1::2]

    # build the mask
    h, w = shape
    mask = np.zeros(h * w, dtype=np.uint8)
    for start, end in zip(run_starts, run_ends):
        mask[start:end] = 1

    # transform the numpy array from flat to the original image shape
    return mask.reshape(shape)


def build_mask(encodings, labels):
    """ takes a pair of lists of encodings and labels, 
        and turns them into a 3d numpy array of shape (256, 1600, 4) 
    """

    # initialise an empty numpy array
    mask = np.zeros((256, 1600, 4), dtype=np.uint8)

    # building the masks
    for rle, label in zip(encodings, labels):
        # classes are [1, 2, 3, 4], corresponding indeces are [0, 1, 2, 3]
        index = label - 1

        # fit the mask into the correct layer
        # note we need to transpose the matrix to account for
        # numpy and openCV handling width and height in reverse order
        mask[:, :, index] = rle_to_mask(rle).T

    return mask
