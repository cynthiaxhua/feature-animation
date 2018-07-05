import matplotlib.pyplot as plt
import matplotlib.patches as patches

colors = ['red', 'blue', 'green']

# Displays the HoG features next to the original image
def plot_img_with_bbox(im, bboxes, title_text = None):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for color, feat in zip(colors, bboxes):
        for _, bbox in bboxes[feat]:
            ax.add_patch(
                patches.Rectangle(
                    (bbox[0], bbox[1]),
                    bbox[2],
                    bbox[3],
                    fill=False,
                    edgecolor=color
                )
            )

    plt.imshow(im, 'gray')
    if title_text is not None:
        plt.title(title_text)

# simple plotting for debugging purposes
def plot_bbox(im, bboxes):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for bbox in bboxes:
        ax.add_patch(
            patches.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2],
                bbox[3],
                fill=False,
                edgecolor='red'
            )
        )

    plt.imshow(im, 'gray')
