# Cat-vs-Dog-Classification
use CNN / DNN to classify the label of the image (cat/dog); image argumentation / transfer learning -> 95% accuracy or above

## Configuration
* tensorflow == 2.7 & python == 3.8.0 (very important as tensorflow > 2.7 changes preprocess api)
* pillow (any version) for image loader
* matplotlib, numpy for data processing and graph

## Setup
### Manually
- Download the cats-vs-dogs zip from https://storage.googleapis.com/mledu-datasets/cats_and_dogs.zip
- Manually delete two dbs in ImagePet/Dogs and ImagePet/Cats
### Linux
- !wget --no-check-certificate \
    "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip", notebook will automatically download it.
- !find /tmp/PetImages/ -type f ! -name "*.jpg" -exec rm {} +' two dbs in the pets image will automatically delete

## Transfer learning 
- Download the model weight data from https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5

### Tips
* Don't recommand run it locally!!!
* Otherwise run it applying GPUs:

pip install tensorflow-gpu  
import tensorflow as tf  
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

