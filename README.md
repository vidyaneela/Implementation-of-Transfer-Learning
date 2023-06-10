# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for CIFAR-10 dataset classification using VGG-19 architecture.
## Problem Statement and Dataset
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.
The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

Here are the classes in the dataset, as well as 10 random images from each:

![237177029-8d4d28a6-7df1-461a-b442-2723754c0e58](https://github.com/vidyaneela/Implementation-of-Transfer-Learning/assets/94169318/0c867e5a-02f9-4e55-b90e-0f50b649c456)

VGG19 is a variant of the VGG model which in short consists of 19 layers (16 convolution layers, 3 Fully connected layer, 5 MaxPool layers and 1 SoftMax layer).
Now we have use transfer learning with the help of VGG-19 architecture and use it to classify the CIFAR-10 Dataset


## DESIGN STEPS
### STEP 1:

Import tensorflow and preprocessing libraries

### STEP 2:

Load CIFAR-10 Dataset & use Image Data Generator to increse the size of dataset

### STEP 3:

Import the VGG-19 as base model & add Dense layers to it

### STEP 4:

Compile and fit the model

### Step 5:

Predict for custom inputs using this model

## PROGRAM
```
Developed by : Vidya Neela M
Reg no : 212221230120
```
Libraries
```
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

(x_train,y_train),(x_test,y_test)=cifar10.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

base_model=VGG19(
    include_top=False,
    weights='imagenet',
    input_shape=(32,32,3)
)

for layer in base_model.layers:
  layer.trainable = False

model=Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(256,activation=('relu')))
model.add(Dropout(.5))
model.add(Dense(10,activation=('softmax')))

model.summary()

model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

model.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_test, y_test), callbacks=[learning_rate_reduction])

import pandas as pd
metrics = pd.DataFrame(model.history.history)

metrics[['loss','val_loss']].plot()
metrics[['accuracy','val_accuracy']].plot()

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
x_test_predictions = np.argmax(model.predict(x_test), axis=1)

print(confusion_matrix(y_test,x_test_predictions))
print(classification_report(y_test,x_test_predictions))


```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot:

![241361141-9fd174bf-f662-4320-8e63-55b8fab65395](https://github.com/vidyaneela/Implementation-of-Transfer-Learning/assets/94169318/9835def3-c786-4566-a7b0-9848c5715182)



### Confusion Matrix

![241361148-0b7f8157-055a-4cae-9311-679e6a62a16f](https://github.com/vidyaneela/Implementation-of-Transfer-Learning/assets/94169318/396ad930-1907-4b1b-8f8e-5d96553e303a)

### Classification Report

![242802088-1d96e20b-f20d-44aa-bf18-33776fdd501f](https://github.com/vidyaneela/Implementation-of-Transfer-Learning/assets/94169318/0b64e413-4f76-4ba3-aa9c-b2d7604bf837)

### Confusion Matrix

![242802157-aaa36f33-0a76-4a09-b422-6ec5919fceff](https://github.com/vidyaneela/Implementation-of-Transfer-Learning/assets/94169318/8589a21f-3b81-4443-a42b-6a0726375891)

### Conclusion:
We got an Accuracy of 60% with this model.There could be several reasons for not achieving higher accuracy. Here are a few possible explanations: Dataset compatibility: VGG19 was originally designed and trained on the ImageNet dataset, which consists of high-resolution images.

In contrast, the CIFAR10 dataset contains low-resolution images (32x32 pixels).

The difference in image sizes and content can affect the transferability of the learned features.

Pretrained models like VGG19 might not be the most suitable choice for CIFAR10 due to this disparity in data characteristics.

### Inadequate training data:
If the CIFAR10 dataset is relatively small, it may not provide enough diverse examples for the model to learn robust representations.

Deep learning models, such as VGG19, typically require large amounts of data to generalize well.

In such cases, you could consider exploring other architectures that are specifically designed for smaller datasets, or you might want to look into techniques like data augmentation or transfer learning from models pretrained on similar datasets.

### Model capacity:
VGG19 is a deep and computationally expensive model with a large number of parameters.

If you are limited by computational resources or working with a smaller dataset, the model's capacity might be excessive for the task at hand.

In such cases, using a smaller model architecture or exploring other lightweight architectures like MobileNet or SqueezeNet could be more suitable and provide better accuracy.

## RESULT:
Thus, transfer Learning for CIFAR-10 dataset classification using VGG-19 architecture is successfully implemented.
