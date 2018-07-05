import numpy as np
np.set_printoptions(linewidth=np.nan)
import matplotlib.pyplot as plt
import cv2
from face_data import load_faces
from plotting import plot_img_with_bbox

def get_bounding_boxes(im, window_size=100, shift_size=50, thresh=0.3):
    H, W = im.shape
    bboxes = {}

    def get_window(j, i):
        return im[j:j+window_size, i:i+window_size]

    for j in range(0, H-window_size, shift_size):
        for i in range(0, W-window_size, shift_size):
            window = get_window(j, i)

            # pred_class, confidence = model.predict(window)
            raise NotImplementedError

            if pred_class == None or \
               confidence < thresh:
                # no bounding box
                continue

            if pred_class:
                if pred_class not in bboxes:
                    bboxes[pred_class] = []
                bbox = (j, i, window_size, window_size)
                bboxes[pred_class].append((confidence, bbox))

    return bboxes

def non_maximal_suppression(bboxes):
    for feat in bboxes:
        if feat == 'eye':
            top_bboxes = sorted(bboxes[feat])[-2:]
        elif feat == 'nose':
            top_bboxes = sorted(bboxes[feat])[-1:]
        elif feat == 'mouth':
            top_bboxes = sorted(bboxes[feat])[-1:]
        elif feat == 'ear':
            top_bboxes = sorted(bboxes[feat])[-2:]

        bboxes[feat] = top_bboxes

    return bboxes

if __name__ == "__main__":
    # get a face image
    faces = load_faces(n=10)

    # for each face
    for face in faces:
        # zero for grayscale
        im = cv2.imread(face, 0)
        cv2.imshow('image', im)
        cv2.waitKey(0)
        
        continue

        # slide a window over the face and get bounding boxes
        bboxes = get_bounding_boxes(im)

        # non-maximal suppression
        bboxes = non_maximal_suppression(bboxes)

        plot_img_with_bbox(im, bboxes)
        plt.show()

    # convert to SVG
    raise NotImplementedError
    # identify SVG components
    raise NotImplementedError
    # animate
    raise NotImplementedError
