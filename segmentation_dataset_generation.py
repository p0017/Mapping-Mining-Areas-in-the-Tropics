import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio.plot
import rasterio.windows
import rasterio.merge
import shapely
import shapely.geometry
import shapely.ops
import random
import os
import skimage
import cv2
import requests
import urllib.request
from argparse import ArgumentParser
import json

from utils import get_bbox, global_to_local_coords, check_if_inside_bbox, replace_at_bbox_borders

#This script is used for the generation of image datasets which can be used for training and inference of segmentation models.
#It reads two polygon datasets, requests satellite imagery of the corresponding locations from Planet,
#and creates a directory containing the satellite images and a directory containing the segmentation masks,
#each split into 0.8/0.1/0.1 train/test/val splits.

#/data
#   /segmentation
#       /mining_polygons_combined.gpkg
#       /2016
#           /img_dir
#               /train
#               /test
#               /val
#           /gpkg
#       /2017
#           ...
#       /2018
#           ...
#       /2019
#           /ann_dir
#               /train
#               /test
#               /val
#           /img_dir
#               /train
#               /test
#               /val
#           /gpkg
#       /2020
#           ...
#       ...
#
#   /tiff_tiles
#       /2016
#       /2017
#       /2018
#       ...

#ann_dir is only required for 2019, since we trained the model on this year.


pd.options.mode.chained_assignment = None

parser = ArgumentParser()
parser.add_argument('-y', '-year', required=True)
#Eight options for year, from '2016' up to '2024'
#If one wants to include more recent data, the corresponding Planet parameter needs to be added to the dict below
args = parser.parse_args()
year = args.y

gdf = gpd.read_file("./data/segmentation/mining_polygons_combined.gpkg")
#Reading the union of two datasets
#We are using the union since they intersect a lot
#Maus, Victor, et al. "An update on global mining land use." Scientific data 9.1 (2022): 1-11.
#https://www.nature.com/articles/s41597-022-01547-4.
#Tang, Liang, and Tim T. Werner. "Global mining footprint mapped from high-resolution satellite imagery." Communications Earth & Environment 4.1 (2023): 134.
#https://www.nature.com/articles/s43247-023-00805-6

countries = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
#Loading a country code dataset
#Will be deprecated sometime in the future

a = gdf['geometry'].apply(lambda x: x.intersects(countries.geometry))
gdf['ISO3_CODE'] = (a * countries['iso_a3']).replace('', np.nan).ffill(axis='columns').iloc[:, -1]
gdf['COUNTRY_NAME'] = (a * countries['name']).replace('', np.nan).ffill(axis='columns').iloc[:, -1]
#Checking in which country the individual polygons are located
gdf['AREA'] = gdf.geometry.area
gdf['id'] = gdf.index

gdf['bbox'] = None #bbox of polygon as shapely polygon object
gdf['tile_ids'] = [np.array([], dtype=object, ndmin=1) for i in gdf.index] #id of tiles on which the polygon is located
gdf['tile_urls'] = [np.array([], dtype=object, ndmin=1) for i in gdf.index] #url of tiles on which the polygon is located
#gdf['tile_ids'] = None
#gdf['tile_urls'] = None
gdf['tile_bboxes'] = None #bboxes of tiles on which the polygon is located
gdf['x_poly'] = None #x coordinates of polygon inside tile coordinate system
gdf['y_poly'] = None #y coordinates of polygon inside tile coordinate system
gdf['x_bbox'] = None #x coordinates of polygon bbox inside tile coordinate system
gdf['y_bbox'] = None #y coordinates of polygon bbox inside tile coordinate system

gdf.reset_index(drop=True, inplace=True)


iso_non_nicfi = ['USA', 'CHN', 'AUS', 'SAU', 'MRT', 'DZA', 'LBA', 'EGY', 'OMN', 'YEM', 'NCL', 'MAR', 'ESH', 'LBY', 'TUN', 'JOR', 'ISR', 'PSE', 'SYR', 'LBN', 'IRQ', 'KWT', 'IRN', 'AFG', 'PAK', 'URY', 'TWN', 'KOR', 'PRK', 'JPN', 'ARE', 'QAT', 'PRI']
nicfi_bbox = gpd.GeoDataFrame(index=[0], crs=4326, geometry=[shapely.Polygon([(-180, 30), (180, 30), (180, -30), (-180, -30), (-180, 30)])])

in_nicfi = gdf.intersects(nicfi_bbox.geometry[0])
gdf_ground_truth = gdf[in_nicfi]

nicfi_subset = [False if iso in iso_non_nicfi else True for iso in gdf['ISO3_CODE']]
gdf = gdf[nicfi_subset]

gdf.reset_index(drop=True, inplace=True)

PLANET_API_KEY = 'PLAK32a5e7b39f75479f8bded1bd73cfc774'
#setup Planet base URL
API_URL = "https://api.planet.com/basemaps/v1/mosaics"
#setup session
session = requests.Session()
#authenticate
session.auth = (PLANET_API_KEY, "")

print()
print('processing', year)

#This is the dict in which one needs to add the corresponding Planet parameters if one wants to include more recent data
#The first parameter is the primary source, the second on is the secondary if a mine is not covered by the primary
nicfi_urls = {'2016':'planet_medres_normalized_analytic_2016-06_2016-11_mosaic',
              '2017':'planet_medres_normalized_analytic_2017-06_2017-11_mosaic',
              '2018':'planet_medres_normalized_analytic_2018-06_2018-11_mosaic',
              '2019':'planet_medres_normalized_analytic_2019-06_2019-11_mosaic',
              '2020':'planet_medres_normalized_analytic_2020-06_2020-08_mosaic',
              '2021':'planet_medres_normalized_analytic_2021-07_mosaic',
              '2022':'planet_medres_normalized_analytic_2022-08_mosaic',
              '2023':'planet_medres_normalized_analytic_2023-06_mosaic',
              '2024':'planet_medres_normalized_analytic_2024-05_mosaic'}

#set params for search using name of primary mosaic
parameters = {"name__is" : nicfi_urls[year]}
#make get request to access mosaic from basemaps API
res = session.get(API_URL, params = parameters)
mosaic = res.json()

#get id
mosaic_id = mosaic['mosaics'][0]['id']

print('requesting tiles')
for j in range(len(gdf)):
    try:
        if gdf['tile_urls'][j].size == 0:
            #getting bboxes of all polygons
            random.seed(int(gdf['id'][j]))
            gdf['bbox'][j] = get_bbox(gdf['geometry'][j])

            #converting bbox to string for search params
            bbox_for_request = list(gdf['bbox'][j].bounds)
            string_bbox = ','.join(map(str, bbox_for_request))

            #search for mosaic tile using AOI
            search_parameters = {
                'bbox': string_bbox,
                'minimal': False
            }
            #accessing tiles using metadata from mosaic
            quads_url = "{}/{}/quads".format(API_URL, mosaic_id)
            res = session.get(quads_url, params=search_parameters, stream=True)

            quads = res.json()
            items = quads['items']

            #getting all required tile ids and urls
            urls = np.array([], dtype=object)
            ids = np.array([], dtype=object)
            bboxes = []

            for i in range(len(items)):
                if 'download' in items[i]['_links'].keys():
                    urls = np.append(urls, items[i]['_links']['download'])
                    ids = np.append(ids, items[i]['id'])
                bboxes.append(items[i]['bbox'])

            gdf['tile_urls'][j] = urls
            gdf['tile_ids'][j] = ids
            gdf['tile_bboxes'][j] = bboxes

    #Sometimes, out-out-bounds tiles are requested
    except json.JSONDecodeError as e:
        print('Requested tile which is not covered by NICFI', e)
        pass

#Planet does not cover a great amount of polygons
no_tiles_found = [False if tile_id.size == 0 else True for tile_id in gdf['tile_ids']]
print('no tiles found for', no_tiles_found.count(False), 'out of', len(gdf), 'polygons')
gdf = gdf[no_tiles_found]
gdf.reset_index(drop=True, inplace=True)


#calculating each polygons position on their own tile/mosaic, we will need those later one by one
for i in range(len(gdf)):
    gdf['x_poly'][i], gdf['y_poly'][i] = global_to_local_coords(gdf['geometry'][i], gdf['tile_bboxes'][i])
    gdf['x_bbox'][i], gdf['y_bbox'][i] = global_to_local_coords(gdf['bbox'][i], gdf['tile_bboxes'][i], is_bbox=True)


def prepare_and_save(gdf:gpd.geodataframe.GeoDataFrame, set_type:str):
    """
    Iterates over all mining polygons in gdf, reads their corresponding .tiff tiles, calculates their postion on these tiles,
    and produces .png images and segmentation masks of size 512x512 for training and prediction.

    Parameters
    -------------

    gdf: A geopandas GeoDataFrame containing mining polygons and infos about their corresponding .tiff tiles.
    type: geopandas.geodataframe.GeoDataFrame
    values: Any.
    default: No default value.

    set_type: The set type which is being processed, only needed for specifying in the right directory.
    type: str
    values: 'train', 'test' or 'val'
    default: No default value.

    Example
    -------------

    prepare_and_save(gdf_train, set_type='train')
    prepare_and_save(gdf_test, set_type='test')
    prepare_and_save(gdf_val, set_type='val')

    """

    print('processing', set_type)
    for k in range(len(gdf)):
        try:
            gdf['tile_urls'] = gdf['tile_urls'].apply(lambda x: np.array(x, ndmin=1))
            gdf['tile_ids'] = gdf['tile_ids'].apply(lambda x: np.array(x, ndmin=1))

            #downloading all required tiles
            for url, id in zip(gdf['tile_urls'][k], gdf['tile_ids'][k]):
                filename = './data/tiff_tiles/{}/{}.tiff'.format(year, id)
                #checks if file already exists
                if not os.path.isfile(filename):
                    urllib.request.urlretrieve(url, filename)

            #getting all secondary polygons which are located on one of the tiles the primary polygon is located on
            #and subsetting the dataset accordingly
            on_same_tile = [any(id in id_list for id in gdf['tile_ids'][k]) for id_list in gdf['tile_ids']]
            gdf_on_same_tile = gdf[on_same_tile]
            gdf_on_same_tile.reset_index(drop=True, inplace=True)



            #tile/mosaic bbox of the primary polygon
            tile_bboxes = gdf['tile_bboxes'][k]

            #calculating the position of all secondary polygons on the tile/mosaic, so we can calculate which are located inside the primary polygons bbox
            for i in range(len(gdf_on_same_tile)):
                x_poly_in_tile, y_poly_in_tile = global_to_local_coords(gdf_on_same_tile['geometry'][i], tile_bboxes)
                gdf_on_same_tile['x_poly'][i] = x_poly_in_tile
                gdf_on_same_tile['y_poly'][i] = y_poly_in_tile

                x_bbox, y_bbox = global_to_local_coords(gdf_on_same_tile['bbox'][i], tile_bboxes, is_bbox=True)
                gdf_on_same_tile['x_bbox'][i] = x_bbox
                gdf_on_same_tile['y_bbox'][i] = y_bbox

            #bbox of the primary polygon and its offset on the tile/mosaic
            x_bbox = gdf['x_bbox'][k]
            y_bbox = gdf['y_bbox'][k]
            x_offset = int(x_bbox[2])
            y_offset = int(y_bbox[0])
            bbox_size = int(x_bbox[1] - x_bbox[3])



            #checking which of the polygons on the tile/mosaic are actually located inside the primary polygons bbox
            #since some polygons are located closely to each other, it can occur that secondary polygons are partly located inside the bbox of the primary polygon
            in_same_bbox = []
            for i in range(len(gdf_on_same_tile)):
                x_poly = gdf_on_same_tile['x_poly'][i].copy()
                y_poly = gdf_on_same_tile['y_poly'][i].copy()
                poly_positions = check_if_inside_bbox(x_poly, y_poly, x_offset, y_offset, bbox_size)

                #we will only count secondary polygons which have more than two points inside the primary polygons bbox
                if poly_positions.count(True) > 2:
                    in_same_bbox.append(True)
                else:
                    in_same_bbox.append(False)

            #and again subsetting the dataset accordingly
            gdf_in_same_bbox = gdf_on_same_tile[in_same_bbox]
            gdf_in_same_bbox.reset_index(drop=True, inplace=True)



            #merging all required tiles into a mosaic
            tile_mosaic = []
            for id in gdf['tile_ids'][k]:
                img = rasterio.open('./data/tiff_tiles/{}/{}.tiff'.format(year, id))
                tile_mosaic.append(img)

            #we have got four color channels, red, green, blue, and NIR
            mosaic, output = rasterio.merge.merge(tile_mosaic, indexes=[1,2,3,4])
            #array needs to be cut according to the primary polygons bbox
            rgb = mosaic[:, y_offset:y_offset+bbox_size, x_offset:x_offset+bbox_size]

            #channels need to be scaled down to 512x512 if needed, using bicubic interpolation
            rgb_resized = []
            for channel in rgb:
                channel_resized = cv2.resize(channel, dsize=(512,512), interpolation=cv2.INTER_CUBIC)
                rgb_resized.append(channel_resized)

            rgb_resized = np.array(rgb_resized).T
            cv2.imwrite('./data/segmentation/{}/img_dir/{}/{}.png'.format(year, set_type, gdf['id'][k]), 255*rgb_resized)

            if year == '2019':
                #turning the polygons into a target array of zeros and ones
                bbox_size = int(gdf['x_bbox'][k][1] - gdf['x_bbox'][k][3])
                target = np.zeros((bbox_size, bbox_size), 'uint8')

                for i in range(len(gdf_in_same_bbox)):
                    x_poly = gdf_in_same_bbox['x_poly'][i].copy()
                    y_poly = gdf_in_same_bbox['y_poly'][i].copy()
                    poly_positions = check_if_inside_bbox(x_poly, y_poly, x_offset, y_offset, bbox_size)

                    if poly_positions.count(True) > 2:
                        x_poly -= x_offset
                        y_poly -= y_offset
                        x_poly, y_poly = replace_at_bbox_borders(x_poly, y_poly, bbox_size, poly_positions)

                        rr, cc = skimage.draw.polygon(y_poly, x_poly, target.shape)
                        target[rr,cc] = 1

                #also downscaling the polygon target arrays to 512x512, using bicubic interpolation
                target_resized = cv2.resize(target, dsize=(512,512), interpolation=cv2.INTER_CUBIC)
                target_resized = np.array(target_resized).T
                cv2.imwrite('./data/segmentation/{}/ann_dir/{}/{}.png'.format(year, set_type, gdf['id'][k]), target_resized)

        #some .tiff files provided by Planet seem to be corrupted and raise an OSError
        except OSError as e:
            print('Caught OSError', e, 'on polygon', k)
            pass

        except FloatingPointError as e:
            print('Caught normalization error caused by empty color channel on polygon', k)
            print(e)
            pass

        except cv2.error:
            print('Caught error caused by empty color channel on polygon', k)
            pass

        except Exception as e:
            print('Caught', e, 'on polygon', k)
            pass


#0.8 train 0.1 val 0.1 test split
test_indices = []
val_indices = []

np.random.seed(2023)

while len(test_indices) < np.round(len(gdf)*0.1):
    r = np.random.randint(0, len(gdf))
    if (r not in test_indices):
        test_indices.append(r)

while len(val_indices) < np.round(len(gdf)*0.1):
    r = np.random.randint(0, len(gdf))
    if ((r not in test_indices) and (r not in val_indices)):
        val_indices.append(r)

train_indices = [i for i in range(len(gdf)) if ((i not in test_indices) and (i not in val_indices))]
print('train', len(train_indices), 'test', len(test_indices), 'val', len(val_indices))

gdf_train = gdf.iloc[train_indices].copy()
gdf_train.reset_index(drop=True, inplace=True)

gdf_test = gdf.iloc[test_indices].copy()
gdf_test.reset_index(drop=True, inplace=True)

gdf_val = gdf.iloc[val_indices].copy()
gdf_val.reset_index(drop=True, inplace=True)


prepare_and_save(gdf_train, set_type='train')
prepare_and_save(gdf_test, set_type='test')
prepare_and_save(gdf_val, set_type='val')
print(year, 'done')
