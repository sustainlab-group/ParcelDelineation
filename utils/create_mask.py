"""
Expects a csv file with parcel id, maxlat, maxlon, minlat, minlon of each satellite image
Expects a json file with the polygons parsed from the shapefile 

Run python utils/create_mask.py [csv_file_path] [json_file_path]
"""

import json
import numpy as np
import sys
import cv2
import csv
import matplotlib.pyplot as plt
from PIL import Image


def read_csv(csv_file):
  grid = dict()
  keys = ['maxlat', 'maxlon', 'minlat', 'minlon']
  with open(csv_file) as f:
    readCSV = csv.reader(f)
    for index, row in enumerate(readCSV, -1):
      if index == -1:
        continue
      if index not in grid:
        grid[index] = dict()
        for key in keys:
          grid[index][key] = 0
      grid[index]['Parcel_id'] = float(row[0])  #SENTINEL
      

      # Sentinel
      grid[index]['maxlat'] = float(row[1])
      grid[index]['maxlon'] = float(row[2])
      grid[index]['minlat'] = float(row[3])
      grid[index]['minlon'] = float(row[4])
      
  return grid

def point_is_in_bounds(point, w, h):
  if point[0] >= 0 and points[0] > w and point[1] >= 0 and point[1] <= h:
    return True
  return False

def scale_coords(shape_size, geom, grid, index, size_m = 450):
  w, h = shape_size
  min_lat, min_lon, max_lat, max_lon = grid[index]['minlat'], grid[index]['minlon'], grid[index]['maxlat'], grid[index]['maxlon']
  x = geom[:,0]
  y = geom[:,1]
  scale_lon = w/(max_lon - min_lon)
  scale_lat = h/(max_lat-min_lat)
  scaled_x = (x - min_lon) * scale_lon # lon-> x, lat->y
  scaled_y = h - ((y - min_lat) * scale_lat)
  if any(val > w for val in scaled_x) or any(val > h for val in scaled_y) or any(val < 0 for val in scaled_x) or any (val < 0 for val in scaled_y):
     return False, np.concatenate([scaled_x[:,None], scaled_y[:,None]],axis=1)
  return True, np.concatenate([scaled_x[:,None], scaled_y[:,None]],axis=1)


base_dir = 'data/sentinel/'

csv_file = base_dir + 'sentinel_locations.csv' if len(sys.argv) < 2 else sys.argv[1]
json_file = base_dir + 'json_files/' + 'pyshp-all-2000-sentinel.json' if len(sys.argv) < 3 else sys.argv[2]  
with open(json_file) as f:
  shp_dict = json.load(f)

shape_size = (224, 224)
grid = read_csv(csv_file)

for index in range(len(grid.keys())):
  parcel_id = grid[index]['Parcel_id']
  polys = []
  for sh_index, sh in enumerate(shp_dict['features']):
    for coord_idx in range(len(sh['geometry']['coordinates'])):
      geom = np.array(sh['geometry']['coordinates'][coord_idx])
      is_in_bounds, geom_fixed = scale_coords(shape_size, geom, grid, index)
      pts = geom_fixed.astype(int)
      polys.append(pts)

  #Creates the binary mask
  mask = np.zeros(shape_size)
  cv2.polylines(mask, polys, True, color=255,thickness=2)

  #Saves the binary mask file
  cv2.imwrite(base_dir + 'masks/image_binary_' + str(int(parcel_id)) + '.png', np.array(mask))

  #Saves the binary mask filled file
  mask = np.zeros(shape_size)
  cv2.fillPoly(mask, polys, color=255)
  cv2.polylines(mask, polys, True, color=0,thickness=2)
  cv2.imwrite(base_dir + 'masks_filled/image_binary_' + str(int(parcel_id)) + '.png', np.array(mask))
  
  #Saves the overlay file
  im_name = base_dir + 'rgb_image/SENTINEL_' + str(int(parcel_id)) +'.jpeg'
  print(im_name)
  orig_image = cv2.imread(im_name)
  cv2.polylines(orig_image, polys, True, color=(255,255,255),thickness=2)
  cv2.imwrite(base_dir + 'overlay/image_overlay_' + str(int(parcel_id)) + '.jpeg', orig_image)
  
