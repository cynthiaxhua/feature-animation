import os
import numpy as np

def load_faces(n=10000):
    filenames = filter(lambda x: x.endswith('.png'),
                       os.listdir('faces'))

    filenames = ['faces/%s' % s for s in filenames]

    # shuffle the filenames
    np.random.shuffle(filenames)

    # return the first n filenames
    return filenames[:n]
