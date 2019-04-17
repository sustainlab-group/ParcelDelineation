import shapefile
from json import dumps
import fiona
from pyproj import Proj#, transform
import pyproj
import numpy as np
import random
import csv
import shapely
from shapely.geometry import Polygon
from shapely.geometry import shape
from functools import partial
from shapely.ops import transform
import matplotlib.pyplot as plt


def read_csv(original, destination, csv_file='./data/img_bbox.csv'):
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
      maxlat = float(row[1])
      maxlon = float(row[2])
      minlat = float(row[3])
      minlon = float(row[4])

      grid[index]['poly'] = shapely.geometry.box(minlat, minlon, maxlat, maxlon) #Polygon([(minlon, minlat), (minlon, maxlat), (maxlon, maxlat),(maxlon, minlat)])
      project = partial(pyproj.transform, original, destination)
      grid[index]['poly'] = transform(project, grid[index]['poly'])
  return grid

def read_centroid(original, destination, centroid_txt): 
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
      maxlat = float(row[1])
      maxlon = float(row[2])
      minlat = float(row[3])
      minlon = float(row[4])

      grid[index]['poly'] = shapely.geometry.box(minlat, minlon, maxlat, maxlon) 
      project = partial(pyproj.transform, original, destination)
      grid[index]['poly'] = transform(project, grid[index]['poly'])
  return grid

def is_in_parcelID_window(chosen_parcel_id, parcel_ids):
  for parcel_id in parcel_ids:
    if abs(chosen_parcel_id - parcel_id) == 0:
      return True
    if parcel_id > chosen_parcel_id + 20:
      return False
  return False

def get_sliding_parcelID_window(grid):
  parcel_ids = set()
  for index in grid:
    parcel_id = int(grid[index]['Parcel_id'])
    for i in range(parcel_id - 20, parcel_id + 20):
      parcel_ids.add(parcel_id)
  print(len(parcel_ids))
  return sorted(list(parcel_ids))

def listit(t):
  return list(map(listit, t)) if isinstance(t, (list, tuple)) else t

def scale_coords(shape_size, geom, grid, index, size_m = 450):
  w, h = shape_size
  min_lat, min_lon, max_lat, max_lon = grid[index]['minlat'], grid[index]['minlon'], grid[index]['maxlat'], grid[index]['maxlon']
  x = geom[:,0]
  y = geom[:,1]
  w = 224
  h = 224
  scale_lon = w/(max_lon - min_lon)
  scale_lat = h/(max_lat-min_lat)
  scaled_x = (x - min_lon) * scale_lon # lon-> x, lat->y
  scaled_y = h - ((y - min_lat) * scale_lat)
  if any(point_is_in_bounds(val, w) for val in scaled_x) and any(point_is_in_bounds(val,h) for val in scaled_y):
    return True
  return False

def check_polygon_in_bounds(poly, grid):
  for index in grid:
    if poly.intersects(grid[index]['poly']) or grid[index]['poly'].contains(poly):
      return True

  return False

def point_is_in_bounds(point, bound):
  if point >= 0 and point <= bound:
    return True
  return False

# read the shapefile
def dump_shp_to_json(shape_file, grid, output_json='./data/pyshp-all-2000-sentinel-new-json'):
  reader = shapefile.Reader(shape_file)
  original = Proj(fiona.open(shape_file).crs)
  print(fiona.open(shape_file).crs)
  destination = Proj('epsg:4326')
  fields = reader.fields[1:]
  field_names = [field[0] for field in fields]
  buffer = []
  index = 0
  num_matched = 0
  #parcel_ids = get_sliding_parcelID_window(grid)

  project = partial(pyproj.transform, original, destination)
  for sr in reader.iterShapeRecords():
    if index % 10000 == 0:
      print('Parsed ', index)
    index += 1
    geom = sr.shape.__geo_interface__
    shp_geom = shape(geom)

    if check_polygon_in_bounds(shp_geom, grid):
      num_matched += 1
      print(int(sr.record[0]))
      atr = dict(zip(field_names, sr.record))
      geom['coordinates'] = listit(geom['coordinates'])
      for index_coord in range(0, len(geom['coordinates'])):
        for counter in range(0,len(geom['coordinates'][index_coord])):
          x, y = geom['coordinates'][index_coord][counter]
          long, lat = pyproj.transform (original , destination, x, y)
          geom['coordinates'][index_coord][counter] = [lat, long] #(long, lat)
      print("Matched:" + str(index))
      print(sr.record[0])
      print("Number matched:", num_matched)
      buffer.append(dict(type="Feature", geometry=geom, properties=atr))
  # write the GeoJSON file
  geojson = open(output_json, "w")
  geojson.write(dumps({"type": "FeatureCollection",\
  "features": buffer}, indent=2) + "\n")
  geojson.close()


csv_file = './data/img_csv.csv'
shape_file = './data/RPG_2-0__SHP_LAMB93_FR-2017_2017-01-01/RPG/1_DONNEES_LIVRAISON_2017/RPG_2-0_SHP_LAMB93_FR-2017/PARCELLES_GRAPHIQUES.shp'
original = Proj(fiona.open(shape_file).crs)
print(original)
destination = Proj('epsg:4326')
reader = shapefile.Reader(shape_file)
grid = read_csv(destination, original, csv_file)
dump_shp_to_json(shape_file, grid)
