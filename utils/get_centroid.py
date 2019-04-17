import json
import random
import numpy as np

def explode(coords):
  for e in coords:
    if isinstance(e, (float, int)):
      yield coords
      break
    else:
      for f in explode(e):
        yield f

def bbox(f):
  x, y = zip(*list(explode(f['geometry']['coordinates'])))
  return min(x), min(y), max(x), max(y)

file_name = './data/pyshp-2000-random-sentinel.json'

with open(file_name) as f:
  geo_dict = json.load(f)

centroids = []
num_im = 2000
size = []
for feature in geo_dict['features']:
  coords = bbox(feature)
  size.append(float(feature['properties']['SURF_PARC']))
  centroid = [feature['properties']['ID_PARCEL'],(coords[0] + coords[2])/2, (coords[1] + coords[3])/2]
  centroids.append(centroid)

print(len(centroids))
print(max(size))
print(min(size))
print(np.mean(size))
print(np.std(size))
with open('./data/centroid-2000-sentinel.txt', 'w') as f:
  f.write('Latitude, Longitude \n')
  for centroid in centroids:
    f.write(str(centroid[2]) + ',' + str(centroid[1]) + "\n")

with open('./data/centroid-2000-withID-sentinel.txt', 'w') as f:
  f.write('Parcel_id, Latitude, Longitude \n')
  for centroid in centroids:
    f.write(str(centroid[0]) + ',' + str(centroid[2]) + ',' + str(centroid[1]) + "\n")
