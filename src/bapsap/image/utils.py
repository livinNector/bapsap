import numpy as np
import skimage.color
import skimage.measure


def binarize(img, threshold=0.1, type=np.uint8):
    """Binarize an image with a threshold"""
    if img.ndim == 3:
        # reduce to gray level if rbg
        img = skimage.color.rgb2gray(img)
    return (img > threshold).astype(type)


def trim(img, background_thresh=0.1, cutoff_type="below"):
    """Trim the backround of the image with the intensity of the background
    Cutoff types :
        below - remove lower intensity pixels
        above - remove higher intensity pixels
    """
    img_bin = binarize(img, background_thresh)

    assert cutoff_type in ("above", "below"), "Cutoff type can only be above or below"
    if cutoff_type == "above":
        img_bin = 1 - img_bin

    img_slice = skimage.measure.regionprops(img_bin)[
        0
    ].slice  # bounding box of the mask
    return img[img_slice]


def reduce_to_size(img, size, func=np.max):
    h, w, *d = img.shape
    rh, rw = size[0] or h, size[1] or w
    ceil = lambda x: np.ceil(x).astype(int)
    block_size = (ceil(h / rh), ceil(w / rw), *([1] * len(d)))
    return skimage.measure.block_reduce(img, block_size, func)


def func_threshold(img, func):
    return img > func(img)


def get_regions(
    image,
    label_image=None,
    intensity_image=None,
    extra_properties=None,
    return_labels=False,
):
    if label_image is None:
        label_image = skimage.measure.label(image)

    regionprops = skimage.measure.regionprops(
        label_image, intensity_image, extra_properties=extra_properties
    )
    if return_labels:
        return regionprops, label_image
    return regionprops


def mask_from_regions(image, regions):
    h, w = image.shape[:2]
    mask = np.zeros((h, w))
    for region in regions:
        mask[region.slice] += region.image
    return mask


def filtered_region_mask(func, image, regions=None, inplace=False):
    if not inplace:
        image = image.copy()
    if regions is None:
        regions = skimage.measure.regionprops(skimage.measure.label(image))
    filtered_regions = filter(func, regions)
    return mask_from_regions(image, filtered_regions)


def apply_regionwise(func, image, regions=None, condition=None, inplace=False):
    if not inplace:
        new_image = image.copy()
    else:
        new_image = image

    if regions is None:
        regions = skimage.measure.regionprops(skimage.measure.label(image))

    for region in regions:
        if condition is None or condition(region):
            new_image[region.slice] = func(image[region.slice])
    return new_image


from sklearn.cluster import KMeans


class ImgKmeans(KMeans):
    """Wrapper on Kmeans for fitting and compressing RBG images with shape (M,N,3).
    Stores the fitted colors in the colors attribute. Applies the transformation on predicted labels
    to the colors when the transform method is called.
    """

    def fit(self, img):
        img = img[..., :3]  # remove the alpha channel if exists
        img_uniques, weights = np.unique(img.reshape(-1, 3), return_counts=True, axis=0)
        # Reshaping as linear array of pixels
        super().fit(img_uniques, sample_weight=weights)

        # used np.unique as a trick for sorting the colors
        self.cluster_centers_ = np.unique(self.cluster_centers_, axis=0)

        # converting it into np.uint8 for efficiency and simplicity
        self.colors = self.cluster_centers_.astype(np.uint8)

        return self

    def predict(self, img):
        img = img[..., :3]  # taking only the rbg channels

        # compressing the image colors in the input image using kmeans
        labels = super().predict(img.reshape(-1, 3)).reshape(img.shape[:2])
        return labels

    def transform(self, img):
        labels = self.predict(img)
        # replacing the color labels with the colors
        return skimage.color.label2rgb(
            labels, colors=self.colors[1:], alpha=1, bg_label=0, bg_color=self.colors[0]
        ).astype(np.uint8)
