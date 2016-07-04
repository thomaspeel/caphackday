import matplotlib.pyplot as plt
from matplotlib import offsetbox
import numpy as np
from sklearn.manifold import TSNE

from hackday_io import get_images_path, load_image


def main():
    with np.load('projections.npz') as file:
        X = file['test']

    tsne = TSNE()
    projected = tsne.fit_transform(X)

    plot_embedding(projected)

    plt.show()


def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)

    for i in range(X.shape[0]):
        c = 'r' if i < 50 else 'b'
        plt.text(X[i, 0], X[i, 1], str(i), fontdict={'weight': 'bold', 'size': 9}, color=c)

    shown_images = np.array([[1., 1.]])  # just something big
    for num, img in enumerate(load_image([path for path in get_images_path() if 'Episode01' in path])):
        if 11 <= num < 418:
            i = num - 11
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(img, cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


if __name__ == '__main__':
    main()
