import os 
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

train="D:\\waste_management\\DATASET\\garbage_classification"
test="D:\\waste_management\\DATASET\\garbage_classification"

train_gen=ImageDataGenerator(
    rescale=1./255,
    rotation_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_gen=ImageDataGenerator(
    rescale=1./255
)

train_data=train_gen.flow_from_directory(
train,
target_size=(224,224),
       batch_size=32, 
       class_mode="categorical"
       )

test_data=test_gen.flow_from_directory(
test,
target_size=(224,224),
 batch_size=32,
 class_mode="categorical"
 )

bmodel=MobileNetV2(weights="imagenet",input_shape=(224,224,3),include_top=False)
bmodel.trainable=False

x=bmodel.output
x=GlobalAveragePooling2D()(x)
x=BatchNormalization()(x)
x=Dense(132,activation="relu")(x)
output=Dense(3,activation="softmax")(x)

modl=Model(inputs=bmodel.inputs,outputs=output)
modl.compile(optimizer="Adam",
             loss="categorical_crossentropy",
             metrics=['acc'])

es=EarlyStopping(monitor="val_loss",patience=5,restore_best_weights="True")
modl.fit(train_data,validation_data=test_data,epochs=2,callbacks=[es])

bmodel.trainable=True
for layers in bmodel.layers[:-31]:
    layers.trainable=False

modl.compile(optimizer=Adam(1e-5),
             loss="categorical_crossentropy",metrics=['acc'])

modl.fit(train_data,validation_data=test_data,epochs=2,callbacks=[es])

modl.save("waste_management.h5")