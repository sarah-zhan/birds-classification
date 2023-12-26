# [Birds Classification](https://www.kaggle.com/datasets/gpiosenka/100-bird-species/data)

- Data set of 525 bird species from Kaggle. 84,635 training images, 2,625 test images(5 images per species) and 2,625 validation images(5 images per species). You can download the data in Kaggle. Since the data files are too large, I have NOT uploaded the dataset here. It includes train/valid/test dataset, a csv file, and a trained model from the owner of the dataset. We would NOT use the model because we will train it on our own.
- Objective: to use tensorflow, keras to train a model using the dataset and we can use our trained model to make a prediction
- I have used [Saturn Cloud](https://saturncloud.io/) free tier for this project
- All details are in `birds-classification-model-training.ipynb`

## EDA
Explore the dataset
```python
# read data file
data = pd.read_csv("birds.csv")
```
![data1](./photos/data1.png)

```python
data.info()
```
![data2](./photos/data2.png)

```python
data["labels"].value_counts()
```
![data3](./photos/data3.png)

```python
# to load an image
path = "./train/ABBOTTS BABBLER"
name = "001.jpg"
fullname = f"{path}/{name}"
img = load_img(fullname, target_size=(299, 299))
```
```python
# convert the image to numpy arry
x = np.array(img)
```

```python
# check the shape of the image
x.shape
```

## Pre-trained convolutional neural networks
- use Xception

```python
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.applications.xception import decode_predictions
# load the model
model = Xception(weights="imagenet", input_shape=(299, 299, 3))
```

```python
# convert all the images to numpy
X = np.array([x])
# preprocess images
X = preprocess_input(X)
pred = model.predict(X)
# decode the prediction
decode_predictions(pred)
```
### Prediction without training the model
![prediction1](./photos/prediction1.png)

## Transfer training
- **Load the data**
```python
train_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_ds = train_gen.flow_from_directory(
    "./train",
    target_size=(150, 150),
    batch_size=32
)
val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_ds = val_gen.flow_from_directory(
    "./valid",
    target_size=(150, 150),
    batch_size=32,
    shuffle=False
)
```

- **Load the base model**
```python
# base model
base_model = Xception(
    weights="imagenet",
    include_top=False,
    input_shape=(150, 150, 3)
)

# freeze
base_model.trainable = False
```

```python
# train the top
inputs = keras.Input(shape=(150, 150, 3))
# 3D
base = base_model(inputs, training=False)
# => 2D
pooling = keras.layers.GlobalAveragePooling2D()
# => vector
vectors = pooling(base)
# add class layer
outputs = keras.layers.Dense(525)(vectors)

model = keras.Model(inputs, outputs)
```

```python
preds = model.predict(X)
# the output does not make sense without training
preds[0]
```

## Train the model
- **turning with different learning rate [0.001, 0.01, 0.1], 0.001 is the BEST**
```python
def make_model(learning_rate=0.01):
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(150, 150, 3)
    )

    base_model.trainable = False

    #########################################

    inputs = keras.Input(shape=(150, 150, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    outputs = keras.layers.Dense(525)(vectors)
    model = keras.Model(inputs, outputs)

    #########################################

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )

    return model
```
![learningrate](photos/learningrate.png)

- **use checkpoint to save the best model during training**
```python
model.save_weights('model_v1.h5', save_format='h5')

checkpoint = keras.callbacks.ModelCheckpoint(
    'xception_v1_{epoch:02d}_{val_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)
```
```python
learning_rate = 0.001

model = make_model(learning_rate=learning_rate)

history = model.fit(
    train_ds,
    epochs=15,
    validation_data=val_ds,
    callbacks=[checkpoint]
)
```
Due to the limited space in github, all the models have not been uploaded.
**The best one is from epoch#13, val_accuracy = 0.8415**
![v1-model](./photos/v1-model.png)

- **adding more layers**
 - add this code to the original make_model function `inner = keras.layers.Dense(size_inner, activation="relu")(vectors)`
 - turning the inner-size layer with [10, 100, 1000]
 - **innser_size=1000 is the best**
 ---
![innersize](photos/innersize.png)

## Regularization and dropout
- Regularizing by freezing a part of the network
- Adding dropout to our model
- Experimenting with different values
- add this code `drop = keras.layers.Dropout(droprate)(inner)`

- **The best droprate is 0.2.**
---
![droprate](photos/droprate.png)

## Data augmentation
- Different data augmentations
- Training a model with augmentations
- Only use it when the accuracy is better. Only added to the train data, you can pick one or a few together. Need to do your own experiment to decide which one to pick.
```python
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=10,
    height_shift_range=10,
    shear_range=10,
    zoom_range=0.1
    vertical_flip=True
)

train_ds = train_gen.flow_from_directory(
    "./train",
    target_size=(150, 150),
    batch_size=32
)
# val dataset keep the same
val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_ds = val_gen.flow_from_directory(
    "./valid",
    target_size=(150, 150),
    batch_size=32,
    shuffle=False
)
```
**After the augmentation, the accuracy is close to what we have before. It does not improve the result.**
![augmentation](photos/augmentation.png)

---
Initially 150x150 size was used to turn the model. After the best parameters are picked, we can train a larger model with the size of 299x299
---

## Train a larger model
- use the same make_model function with
```python
learning_rate = 0.0005
size = 100
droprate = 0.2
input_size = 299
```
- use checkpoint to save the best model
![v4-model](photos/v4-model.png)
**`xception_v4_36_0.929.h5` is the best**
![largermodel](photos/largermodel.png)

## Use the model the test
```python
# load the model
model = keras.models.load_model("xception_v4_36_0.929.h5")
# evaludate the model - accuracy: 0.9345
model.evaluate(test_ds)
```
# choose an image for the test
```python
path = "./test/ALBATROSS/1.jpg"
img = load_img(path, target_size=(299, 299))
x = np.array(img)
X = np.array([x])
X = preprocess_input(X)
pred = model.predict(X)
classes = list(test_ds.class_indices.keys()) #classes list
class_pred_dict = dict(zip(classes, pred[0]))
max_class = max(class_pred_dict, key=class_pred_dict.get) # => 'ALBATROSS', it matches the real result.
```

---
Since the model is large, when you git the file, an error might prompt<br>
`this exceeds GitHub's file size limit of 100.00 MB remote: error: GH001: Large files detected. You may want to try Git Large File Storage - https://git-lfs.github.com.`

Solution:
- download from https://git-lfs.github.com
- use the below command one by one
```python
git lfs version
git lfs install
git lfs track "*.h5"
git add --all
git commit -m "Add large .h5 files"
git push
```
- if you still have a problem, try
```python
git lfs migrate info
git lfs migrate import --include="*.h5" --include-ref=refs/heads/main
git push origin main
```
---

## Convert Keras to TF-Lite - only include inference (prediction)
- use TF-Lite to do the prediction

```python
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model("xception_v4_36_0.929.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('xception_v4_36_0.929.tflite', 'wb') as f_out:
    f_out.write(tflite_model)
```
- install the packages
```python
!pip install keras-image-helper
!pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
```
- load the model
```python
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model("xception_v4_36_0.929.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('xception_v4_36_0.929.tflite', 'wb') as f_out:
    f_out.write(tflite_model)
```
- get the classes
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_ds = train_gen.flow_from_directory(
    "./train",
    target_size=(150, 150),
    batch_size=32
)
classes = list(train_ds.class_indices.keys())
```
- get predictions
```python
import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor
interpreter = tflite.Interpreter(model_path="xception_v4_36_0.929.tflite")
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

preprocessor = create_preprocessor("xception", target_size=(299, 299))
path = "./test/AFRICAN CROWNED CRANE/1.jpg"
X = preprocessor.from_path(path)
interpreter.set_tensor(input_index, X)
interpreter.invoke()
preds = interpreter.get_tensor(output_index)
class_pred_dict = dict(zip(classes, preds[0]))
max_class = max(class_pred_dict, key=class_pred_dict.get)
```

