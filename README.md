# ParcelDelineation

# Training models

The script contains various parameters you can tweak. You can specify to use a dilated UNet (is_dilated), pretrained UNet (is_imagenet), stacked with 9-images (is_stacked), etc. By default, the model uses a pretrained stacked U-Net model.

```
python train_unet.py 
```

# Evaluation 

Spits out prediction numpy array of model. Also does evaluation of F1 score and accuracy. Expects a csv file that has columns image, mask. 'image' refers to path of the image and 'mask' refers to the path of the binary mask. See more in dataset preparation section.

```
python predict_model.py [path to saved model] [path to csv containing paths to images (e.g. test/validation dataframe)]
```

Does evaluation based on the numpy array paths of the model. 

```
python evaluate_model.py [path to array of predictions] [path to test/validation dataframe csv]
```


# Dataset preparation:

1. Download the French polygons dataset from 

https://www.data.gouv.fr/en/datasets/registre-parcellaire-graphique-rpg-contours-des-parcelles-et-ilots-culturaux-et-leur-groupe-de-cultures-majoritaire/

2. Unzip the dataset under ./data directory

3. Samples random polygons. The number of polygons can be set in the script. Default is set to 2000.
```
python utils/sample_shp.py   
```

4. Reads the shape file to gets the centroid of each polygon (Used as getting coordinates for getting satellite images such as SENTINEL-2). The centroids are the center of the satellite images.
```
python utils/get_centroid.py   
```

5. (a) From step 5, you can extract your own satellite imagery from a public dataset (such as SENTINEL-2 or Digital Globe) and prepare a csv file containing max lat, max lon, min lat, min lon of the image. For satellite images (which comes in tfrecord format for our dataset), the following script extracts jpegs and also the csv file for the max lat, max lon, min lat, min lon of the image, useful for the next step to overlay the polygons onto image. The max lat, max lon, min lat, min lon will specify how large each satellite image spans in size. 
```
python convert_tfrecords_jpeg.py
```

5. (b) Gets only polygons that overlap in bounds of extracted images (Requires a csv-file with a unique parcel identifier, max lat, max lon, min lat, min long of each extracted image).

```
python utils/shp2geo.py
```

6. Creates the masks (boundary and filled) of the extracted polygons and images (File input/output paths are specified in script)

```
python utils/create_mask.py
```

7. Splits data into train/test/val

```
python utils/split_data.py
```


Reference Code:

Unet pretrained models:
https://github.com/qubvel/segmentation_models

Unet keras model:
https://github.com/zhixuhao/unet

Deeplabv3 (beta):
https://github.com/tensorflow/models/tree/master/research/deeplab
