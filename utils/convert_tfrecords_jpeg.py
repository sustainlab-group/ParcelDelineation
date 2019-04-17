import os
import subprocess
import tensorflow as tf
from pprint import pprint
import scipy.misc
import numpy as np
import sys
import csv 

options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
root = "data/"

filenames = ["./data/sentinel_tf/" + f for f in os.listdir("./data/sentinel_tf/")]
print(filenames)
for f in filenames:
    f_sub = f.split('.')[1][10:]
    print(f_sub)
    output_directory = root + "/sentinel/" + f_sub
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    satellite_features = ['RED', 'GREEN', 'BLUE']

    print(">>>>>> Processing: " + f)
    iterator = tf.python_io.tf_record_iterator(f, options=options)
    n = 0
    print(f)
    iter = 10
    csv_file = output_directory  + "/location.json"
    csv_file_keys = ['Parcel_id', 'max_lat', 'max_lon', 'min_lat', 'min_lon']
    with open(csv_file, 'w') as f_csv:
        csv_writer = csv.writer(f_csv)
        csv_writer.writerow(csv_file_keys)
    while iter > 0:
        with open(csv_file, 'a') as f_csv:
            csv_writer = csv.writer(f_csv)
            try:
                record_str = next(iterator)
                ex = tf.train.Example.FromString(record_str)
            #    print(ex.features.feature['Parcel_id'])
                min_lon = min(ex.features.feature['LON'].float_list.value)
                max_lon = max(ex.features.feature['LON'].float_list.value)
                min_lat = min(ex.features.feature['LAT'].float_list.value)
                max_lat = max(ex.features.feature['LAT'].float_list.value)
                idx = int(ex.features.feature['Parcel_id'].float_list.value[0])
                features = []
                for satellite_feature in satellite_features:
                    feature = (ex.features.feature[satellite_feature].float_list.value)
                    feature = np.array(feature)
                    feature = feature.reshape((225, 225, 1))
                    feature = np.flip(feature, axis=0)
                    features.append(feature)

                csv_writer.writerow([idx, max_lat, max_lon, min_lat, min_lon])
                image = np.concatenate(features, axis=2)
                image = image[:224, :224, :]

                if idx != -1:
                    jpeg_path = output_directory + '/' + str(idx) + '.jpeg'
                    scipy.misc.imsave(jpeg_path, image)
          
                    #scipy.misc.toimage(image, cmin=0.0, cmax=...).save(jpeg_path)
                    #writer = tf.python_io.TFRecordWriter(tfrecord_path, options=options)
                    #writer.write(ex.SerializeToString())
                    #writer.close()
                #print(idx)
                n += 1
                if n%10==0:
                    print("       Processed " + str(n) + " records in " + f)
            except:
                iter -= 1
                print(">>>>>> Processed " + str(n) + " records in " + f)

