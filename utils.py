import numpy as np
import pandas as pd
import shapely
import shapely.geometry
import shapely.ops
from scipy.ndimage import binary_erosion, binary_opening, binary_fill_holes

#This script contains some handy functions which are imported into segmentation_dataset_generation.py and gpkg_dataset_generation.py.


def get_bbox(polygon:shapely.geometry.polygon.Polygon) -> shapely.geometry.polygon.Polygon:
    """
    Returns a square bounding box for a given polygon.

    Parameters
    -------------

    polygon: A shapely polygon for which a bounding box should be generated.
    type: shapely.geometry.polygon.Polygon
    values: Any.
    default: No default value.

    Example
    -------------

    import get_bbox from utils
    my_bbox = get_bbox(my_polygon)

    """

    minx, miny, maxx, maxy = polygon.bounds
    #Using three times the max side length of the polygon for the bbox side length
    max_len = max([maxx-minx, maxy-miny])
    x_centroid = (maxx+minx)/2
    y_centroid = (maxy+miny)/2

    #centroid = [random.uniform(x_centroid - max_len/4, x_centroid + max_len/4), random.uniform(y_centroid - max_len/4, y_centroid + max_len/4)]
    centroid = [x_centroid, y_centroid]
    #return shapely.Point(centroid).buffer(distance=random.uniform(3*max_len/4, max_len), quad_segs=1, cap_style='square')
    return shapely.Point(centroid).buffer(distance=max_len, quad_segs=1, cap_style='square')



def count_tiles(tile_bboxes:list) -> (int, int):
    """
    Since some mines stretch along multiple tiles, these tiles must be stitched to gether into a mosaic.
    This function returns a rectangular bounding box for this mosaic and returns how many tiles high and wide this bounding box is.

    Parameters
    -------------

    tile_bboxes: A list of bounding boxes of the individual tiles.
    type: list
    values: Any.
    default: No default value.

    Example
    -------------

    import count_tiles from utils
    my_mosaic_bbox, width, height = count_tiles(my_tile_bboxes)

    """

    mosaic_bbox = tile_bboxes[0].copy()
    x_tile_counter = 1
    y_tile_counter = 1

    #if the polygon is located on more than one tile
    if len(tile_bboxes) > 1:
        #calculating the bounding box of the mosaic composed by the individual tiles
        #and noting the tile positioning
        for bbox in tile_bboxes[1:]:
            #x+
            if bbox[0] < mosaic_bbox[0]: 
                mosaic_bbox[0] = bbox[0]
                x_tile_counter += 1
            #y+
            if bbox[1] < mosaic_bbox[1]: 
                mosaic_bbox[1] = bbox[1]
                y_tile_counter += 1
            #x-
            if bbox[2] > mosaic_bbox[2]: 
                mosaic_bbox[2] = bbox[2]
                x_tile_counter += 1
            #y-
            if bbox[3] > mosaic_bbox[3]: 
                mosaic_bbox[3] = bbox[3]
                y_tile_counter += 1

    return mosaic_bbox, x_tile_counter, y_tile_counter


def global_to_local_coords(poly:shapely.geometry.polygon.Polygon, tile_bboxes:list, is_bbox:bool=False) -> np.ndarray:
    """
    Turns a shapely polygon into two lists of x and y coordinates.
    These new x and y coordinates exist inside the coordinate system the tile/mosaic the polygon is located on.
    This shapely polygon can either be the polgon of a mine or its bounding box.
    For bounding boxes, the coordinates are also corrected, so their size is an integer mutiple of 512.

    Parameters
    -------------

    poly: A shapely polygon of either a mine or its bounding box.
    type: shapely.geometry.polygon.Polygon
    values: Any.
    default: No default value.

    tile_bboxes: A list of bounding boxes of the individual tiles that this polygon lies upon.
    type: list
    values: Any.
    default: No default value.

    is_bbox: States if poly is a mine or a bounding box of a mine.
    type: bool
    values: Any.
    default: False

    Example
    -------------

    import global_to_local_coords from utils
    x, y = global_to_local_coords(my_mining_polygon, my_tile_bboxes, False)

    """

    mosaic_bbox, x_tile_counter, y_tile_counter = count_tiles(tile_bboxes)

    tile_size_x = (mosaic_bbox[2] - mosaic_bbox[0])
    tile_size_y = (mosaic_bbox[3] - mosaic_bbox[1])
    poly = np.array(poly.exterior.coords)

    #moving from a global coordinate system to the tile/mosaic coordinate system
    for i in range(len(poly)):
        #removing the offset and rescaling it according to the number of tiles forming the mosaic
        #and upscaling it to the next greatest multiple of 512 is its a bbox
        poly[i][0] -= mosaic_bbox[0]
        poly[i][0] = np.round((poly[i][0]/tile_size_x) * 4096 * x_tile_counter)

        poly[i][1] -= mosaic_bbox[1]
        poly[i][1] = np.round((poly[i][1]/tile_size_y) * 4096 * y_tile_counter)

    #y coordinates count from top to bottom
    for i in range(len(poly)):
        poly[i][1] = 4096 * y_tile_counter - poly[i][1]

    #bbox sizes need to be a multiple of 512, so they can be scaled down to 512x512 if needed
    if is_bbox:
        bbox_size = int(poly[1][0] - poly[3][0])
        delta = ((bbox_size // 512) + 1) * 512 - bbox_size

        #adding the needed extra space
        poly[0][0] = poly[1][0] = poly[4][0] = poly[0][0] + delta
        poly[1][1] = poly[2][1] = poly[1][1] + delta

    poly_x, poly_y = zip(*poly)

    return np.array(poly_x), np.array(poly_y)



def check_if_inside_bbox(x:list, y:list, x_offset:int, y_offset:int, bbox_size:int) -> list:
    """
    Checks if polygon points are located inside a certain square bounding box.
    Returns a list with a boolean value corresponding to each of those polygon points.

    Parameters
    -------------

    x: A list of x coordinates.
    type: list
    values: Any.
    default: No default value.

    y: A list of y coordinates.
    type: list
    values: Any.
    default: No default value.

    x_offset: The x axis offset of this bounding box in the same coordinate system as the x and y coordinates.
    type: int
    values: Any.
    default: No default value.

    y_offset: The y axis offset of this bounding box in the same coordinate system as the x and y coordinates.
    type: int
    values: Any.
    default: No default value.

    bbox_size: The sidelength of this square bounding box.
    type: int
    values: Any.
    default: No default value.

    Example
    -------------

    import check_if_inside_bbox from utils
    positions = check_if_inside_bbox([123, 102, 99], [25, 36, 46], 50, 30, 40)

    """

    poly_positions = []

    for x_poly, y_poly in zip(x, y):
        #checking if point is inside bbox
        if (x_poly >= x_offset) and (x_poly <= x_offset+bbox_size) and (y_poly >= y_offset) and (y_poly <= y_offset+bbox_size):
            poly_positions.append(True)
        else:
            poly_positions.append(False)

    return poly_positions



def replace_at_bbox_borders(x:list, y:list, bbox_size:int, poly_positions:list) -> (list, list):
    """
    Takes x and y coordinates of a polygon, a square bounding box and a list of boolean values corresponding to these polygon coordinates which specify if they are inside the bounding box.
    And returns new x and y coordinates where every point outside the bounding box is replaced by the closest bounding box border point so that every point is inside of it.
    The boolean list poly_positions can be generated by calling the check_if_inside_bbox function.

    Parameters
    -------------

    x: A list of x coordinates.
    type: list
    values: Any.
    default: No default value.

    y: A list of y coordinates.
    type: list
    values: Any.
    default: No default value.

    bbox_size: The sidelength of this square bounding box.
    type: int
    values: Any.
    default: No default value.

    poly_positions: A list of boolean values specifying if the polygon coordinates are inside the bounding box.
    type: int
    values: Any.
    default: No default value.

    Example
    -------------

    import check_if_inside_bbox, replace_at_bbox_borders from utils
    positions = check_if_inside_bbox(my_x, my_y, my_bbox_x_offset, my_bbox_y_offset, my_bbox_size)
    new_x, new_y = replace_at_bbox_borders(my_x, my_y, my_bbox_size, positions)

    """
    
    new_x = x.copy()
    new_y = y.copy()

    for i in range(len(poly_positions)):
        #only takes poly points which are outside the bbox
        if not poly_positions[i]:
            if new_x[i] < 0:
                new_x[i] = 0
            elif new_x[i] > bbox_size:
                new_x[i] = bbox_size

            if new_y[i] < 0:
                new_y[i] = 0
            elif new_y[i] > bbox_size:
                new_y[i] = bbox_size

    return new_x, new_y


def isnan(a):
    """
    Checks if the object, which can be a series or no series is nan.
    Used for checking invalid bounding boxes.

    Parameters
    -------------

    a: The object to be checked.
    type: Not specified.
    values: Any.
    default: No default value.

    Example
    -------------

    import isnan from utils
    res1 = isnan(np.array([0,1,2]))
    res2 = isnan(0)

    """
    if type(a) == np.ndarray:
        return pd.isnull(a).any()
    else:
        return pd.isnull(a)

#optional postprocessing using morphological operations
def postprocess(pred: np.ndarray, opening_iter:int=1, erosion_iter:int=2) -> np.ndarray:
    """
    Does postprocessing using morphological operations on a 2d Numpy array containing binary values.
    Used on the predictions of the segmentation model.

    Parameters
    -------------

    pred: A 2d Numpy array containing either 0 or 1.
    type: np.ndarray
    values: Any.
    default: No default value.

    opening_iter: The amount of times binary opening is applied.
    type: int
    values: Any.
    default: 1

    erosion_iter: The amount of times binary erosion is applied.
    type: int
    values: Any.
    default: 2

    Example
    -------------

    import postprocess from utils
    postprocessed_prediction = postprocess(prediction, 1, 3)

    """

    pred = binary_fill_holes(pred)
    pred = binary_opening(pred, iterations=opening_iter)
    pred = binary_erosion(pred, iterations=erosion_iter)
    return pred.astype(int)

def close_holes(poly: shapely.Polygon) -> shapely.Polygon:
    """
    Closes all holes inside a shapely polygon if there are any.

    Parameters
    -------------

    poly: A shapely polygon which may contain holes.
    type: shapely.Polygon
    values: Any.
    default: No default value.

    Example
    -------------

    import close_holes from utils
    postprocessed_polygon = close_holes(polygon)

    """

    if poly.interiors:
        return shapely.Polygon(list(poly.exterior.coords))
    else:
        return poly