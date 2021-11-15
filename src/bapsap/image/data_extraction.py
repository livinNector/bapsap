import numpy as np

# Image processing tools
from skimage import io, transform, util, measure, morphology, filters
from skimage.color import rgb2gray, rgb2hsv, label2rgb
from scipy import ndimage

# Image processing utility functions
from . import utils

# For clustering similar balls
from sklearn.cluster import KMeans

# To generate plots of the intermediate steps
import matplotlib.pyplot as plt


def _find_approx_tube_vertical_margins(img, reduction_size=20):
    """
    Returns the approximate Vertical margins of the tubes.
        Uses a binary thersholded image.
    """
    # Reducing the image for efficiency
    h, w = img.shape
    reduced = utils.reduce_to_size(
        img,
        size=(reduction_size, 1),
        func=lambda x, axis: np.max(x, axis=axis) - np.min(x, axis=axis),
    )
    reduced = utils.func_threshold(reduced, filters.threshold_otsu)

    r_h = reduced.shape[0]
    mid_y = r_h // 2
    oct_y = mid_y // 4

    zero_idxs = np.argwhere(reduced.ravel() == 0)
    a = int(max(zero_idxs[zero_idxs < mid_y - oct_y]) * h // reduction_size)
    b = int(min(zero_idxs[zero_idxs > mid_y + oct_y] + 1) * h // reduction_size)

    return slice(a, b)


def extract_ball_color_codes(
    image_file_path,
    n_balls_per_color=4,
    return_tube_coords=False,
    plot=False,
):
    """
    Returns the Coordinates of the tube centers and color code representation
    of the tube state.

    Parameters
    ----------
    image_file_path : str
        Path of the image file of the game screenshot.

    n_balls_per_color : int
        Number of balls in each color, default is 4

    return_tube_coords : bool
        Whether to return the coordinates of the tubes centers

    Returns
    -------
    ball_color_codes : list of lists of ints
        The color codes of the balls as a list of lists of integers,
        in which each list represents a tube and the the values in each tube
        represents code for a given ball. The number of uniques colors k is
        found using the assumption which ranges from 0 - (k-1),
        where k is the number of unique balls. k is found using the assumptiona  Empty tubes are represented by empty lists.

    tube_center_coords : list of coordinates, optional
        The coordinates of the centers of the tubes as a list in the form
        [(y_1,x_1), (y_2,x_2), ..., (y_n,x_n)] for n tubes.

    Examples
    --------
    >>> extract_ball_color_codes("image.png")
    [[1, 5, 2, 3], [2, 5, 4, 1], [4, 4, 2, 5], [5, 1, 1, 3], [4, 3, 3, 2], [], []]

    >>> extract_ball_color_codes("image.png",return_tube_coords=True)
    [[1, 5, 2, 3], [2, 5, 4, 1], [4, 4, 2, 5], [5, 1, 1, 3], [4, 3, 3, 2], [], []]
    **incomplete**
    """

    image = util.img_as_ubyte(io.imread(image_file_path)[..., :3])
    image_gray = util.img_as_ubyte(rgb2gray(image))

    # Initial approximate cropping for efficiency

    v_margins = _find_approx_tube_vertical_margins(image_gray, 20)
    img_cropped = image[v_margins]
    del image
    del image_gray
    h, w = img_cropped.shape[:2]

    # For Coordinate shifting
    y1, x1 = v_margins.start, 0

    def absolute_position(relative_position):
        y, x = relative_position
        return int(y + y1), int(x + x1)

    # image converted to hsv to ger  clear picture of edges
    img_hsv_cropped = rgb2hsv(img_cropped)

    # reducing hsv space to a mono channel
    img_hsv_gray_cropped = util.img_as_ubyte(img_hsv_cropped[..., 1:].dot([0.25, 0.75]))
    
    del img_cropped
    del img_hsv_cropped
    # Filtering edges using sobel filter
    img_edges = filters.sobel(img_hsv_gray_cropped)
    img_edges_bin = (img_edges > 0.03).astype(np.uint8)

    def height(region_mask):
        return region_mask.shape[0]

    def width(region_mask):
        return region_mask.shape[1]

    def ball_regions_filter(region):
        rh, rw = region.height, region.width
        return h // 20 < rh < h // 4 and h // 20 < rw < h // 4

    edge_regions = utils.get_regions(img_edges_bin, extra_properties=(height, width))

    # filtered ball regions from edges
    # This works best for balls containing patterns
    ball_regions = list(filter(ball_regions_filter, edge_regions))

    # convex hull of filtered balls from edges
    balls_filtered = utils.apply_regionwise(
        morphology.convex_hull_image,
        utils.mask_from_regions(img_edges_bin, ball_regions),
        ball_regions,
    )

    # Inverting endges to obtain inner regions
    # this  works best for plain colored balls
    img_edges_inverted = morphology.binary_erosion(
        1 - img_edges_bin, ndimage.generate_binary_structure(2, 2)
    )

    # regions from inverted edges
    inv_edge_regions = utils.get_regions(
        img_edges_inverted, extra_properties=(height, width)
    )

    inv_ball_regions = list(filter(ball_regions_filter, inv_edge_regions))

    # Balls mask using inverted edge regions
    inv_balls_filtered = utils.apply_regionwise(
        morphology.convex_hull_image,
        utils.mask_from_regions(img_edges_inverted, inv_ball_regions),
        inv_ball_regions,
    )

    def combine_regions(edge_region, inv_edge_region, plot=True):
        prod = edge_region * inv_edge_region
        if np.mean(prod) > 0.1:
            return prod
        return np.logical_or(edge_region, inv_edge_region)

    # combining the regions obtained from both
    # edge regiosn and inverted edge regions
    img_balls = combine_regions(balls_filtered, inv_balls_filtered).astype(np.uint8)
    img_ball_regions, img_ball_labels = utils.get_regions(
        img_balls, intensity_image=img_hsv_gray_cropped, return_labels=True
    )

    if plot:
        plt.imsave(
            "predicted_labels.png",
            label2rgb(
                img_ball_labels,
                image=img_hsv_gray_cropped * img_balls,
                bg_label=0,
                alpha=0.4,
            ),
        )
    # Mask of Tube regions
    img_tubes = morphology.binary_opening(
        ndimage.binary_fill_holes(img_edges_bin), morphology.disk(5)
    ).astype(np.uint8)

    img_tube_regions = utils.get_regions(img_tubes)

    tube_coords = [absolute_position(region.centroid) for region in img_tube_regions]

    def get_ball_features(image):
        image = transform.resize(image, (32, 32))
        h, v = 3, 3
        reduced = utils.reduce_to_size(image, (h, v), np.mean)[:h, :v, ...]
        return reduced.flatten()

    ball_features = np.array(
        [
            np.array(get_ball_features(region.intensity_image.copy()))
            for region in img_ball_regions
        ]
    )

    def get_model(X):
        m = X.shape[0]
        k = m // n_balls_per_color
        return KMeans(k).fit(X)

    img_model = get_model(ball_features)
    ball_color_labels = img_model.labels_ + 1  # Kmeans labels starts from 0
    img_ball_color_labels = img_balls.copy()

    for region, label in zip(img_ball_regions, ball_color_labels):
        img_ball_color_labels[region.slice][region.image] = label

    ball_color_codes = []
    for region in img_tube_regions:
        tube_ball_regions = measure.regionprops(img_ball_labels[region.slice])
        if not tube_ball_regions:
            ball_color_codes.append([])
            continue

        ball_color_codes.append(
            [
                img_ball_color_labels[region.slice][
                    tuple(map(int, ball_region.centroid))
                ]
                for ball_region in tube_ball_regions[::-1]
                # region props get labels from top to bottom
            ]
        )
    if return_tube_coords:
        return ball_color_codes, tube_coords
    return ball_color_codes
