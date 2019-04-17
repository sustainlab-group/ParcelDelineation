import shapefile
from json import dumps
import fiona
from pyproj import Proj
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


def listit(t):
  return list(map(listit, t)) if isinstance(t, (list, tuple)) else t

def dump_shp_to_json_random(shape_file, out_file='pyshp-2000-random-sentinel.json', num_random=2000):
  reader = shapefile.Reader(shape_file)
  original = Proj(fiona.open(shape_file).crs)
  print(fiona.open(shape_file).crs)
  destination = Proj('epsg:4326')
  fields = reader.fields[1:]
  field_names = [field[0] for field in fields]
  buffer = []
  index = 0
  rand_indices = np.random.choice(len(reader), num_random) 
  print(rand_indices)
  for i in rand_indices:
    sr = reader.shapeRecord(i)
    atr = dict(zip(field_names, sr.record))
    geom = sr.shape.__geo_interface__
    geom['coordinates'] = listit(geom['coordinates'])
    for index_coord in range(0, len(geom['coordinates'])):
      for counter in range(0,len(geom['coordinates'][index_coord])):
        x, y = geom['coordinates'][index_coord][counter]
        long, lat = pyproj.transform (original , destination, x, y)
        geom['coordinates'][index_coord][counter] = [lat, long] #(long, lat)
    buffer.append(dict(type="Feature", geometry=geom, properties=atr))
      # write the GeoJSON file
  geojson = open(out_file, "w")
  geojson.write(dumps({"type": "FeatureCollection",\
  "features": buffer}, indent=2) + "\n")
  geojson.close()


shape_file = './data/RPG_2-0__SHP_LAMB93_FR-2017_2017-01-01/RPG/1_DONNEES_LIVRAISON_2017/RPG_2-0_SHP_LAMB93_FR-2017/PARCELLES_GRAPHIQUES.shp'
dump_shp_to_json_random(shape_file, './data/pyshp-2000-random-sentinel.json')
