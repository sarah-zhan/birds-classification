# [Birds Classification](https://www.kaggle.com/datasets/gpiosenka/100-bird-species/data)

- Data set of 525 bird species from Kaggle. 84635 training images, 2625 test images(5 images per species) and 2625 validation images(5 images per species). You can download the data in Kaggle. Since the file is too large, I have NOT uploaded the archive.zip of the dataset here. It includes train/valid/test dataset, a csv file, and a trained model from the owner of the dataset. We would NOT use the model because we will train it on our own.
- Objective: to use tensorflow, keras to train a model using the dataset and we can use our trained model to make a prediction

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
**Prediction without training the model**
![prediction1](./photos/prediction1.png)

## Transfer training
- **load the data**
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
- **turning with different learning rate [0.001, 0.01, 0.1]**
![learningrate](photos/learningrate.png)

- 
