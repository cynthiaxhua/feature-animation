import numpy as np
np.set_printoptions(linewidth=np.nan)
import matplotlib.pyplot as plt
import cv2
from face_data import load_faces
from plotting import plot_img_with_bbox
from conversion import get_window_3_stroke
from neuralnet import FeatureClassifier

# property of the neural net classifier
MAX_LEN = 125
labels = ['ear', 'eye', 'mouth', 'nose']

def get_bounding_boxes(im,
                       model,
                       window_size=100,
                       shift_size=50,
                       scale_factor=10,
                       thresh=0.5,
                       verbose=False):
    """
    params:
        im              image (256x256)
        model           Feature Classifier
        window_size     size of bounding box
        shift_size      size of window shift
        scale_factor    scale of strokes
        thresh          minimum confidence
    """
    H, W = im.shape
    bboxes = {}

    shape = (window_size, window_size)
    for j in range(0, H-window_size, shift_size):
        for i in range(0, W-window_size, shift_size):
            stroke = get_window_3_stroke(im, j, i,
                                         window_shape=shape,
                                         scale_factor=scale_factor)

            pred_class, confidence = model.predict(stroke)
            if verbose:
                print pred_class, confidence
                model.draw(stroke)

            if pred_class == None:
                if verbose:
                    print 'no classification'
                continue

            if pred_class not in bboxes:
                bboxes[pred_class] = []
            bbox = (j, i, window_size, window_size)
            bboxes[pred_class].append((confidence, bbox))

    return bboxes

def non_maximal_suppression(bboxes, eyes=2, noses=1, mouths=1):
    for feat in bboxes:
        if feat == 'eye':
            top_bboxes = sorted(bboxes[feat])[-eyes:]
        elif feat == 'nose':
            top_bboxes = sorted(bboxes[feat])[-noses:]
        elif feat == 'mouth':
            top_bboxes = sorted(bboxes[feat])[-mouths:]

        bboxes[feat] = top_bboxes

    return bboxes

def classify(im, window_size=100, shift_size=25, scale_factor=10, thresh=0.5,
             eyes=2, noses=1, mouths=1,
             verbose=False):

    model = FeatureClassifier()
    # slide a window over the face and get bounding boxes
    bboxes = get_bounding_boxes(im, model,
                                window_size=window_size,
                                shift_size=shift_size,
                                scale_factor=scale_factor,
                                thresh=thresh,
                                verbose=verbose)

    # non-maximal suppression
    bboxes = non_maximal_suppression(bboxes, eyes, noses, mouths)

    # make plots
    plot_img_with_bbox(im, bboxes)
    plt.show()

if __name__ == "__main__":
    # get a face image
    faces = load_faces(n=10)

    model = FeatureClassifier()

    # for each face
    for face in faces:
        # zero for grayscale
        im = cv2.imread(face, 0)
        cv2.imshow('image', im)
        cv2.waitKey(0)

        classify(im)

    # convert to SVG
    raise NotImplementedError
    # identify SVG components
    raise NotImplementedError
    # animate
    raise NotImplementedError
