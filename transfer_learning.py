from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications.vgg16 import preprocess_input
import cv2
import numpy as np

img_width, img_height = 224, 224
train_data_dir = "data/train"
validation_data_dir = "data/validation"
nb_train_samples = 300
nb_validation_samples = 100
batch_size = 16
epochs = 1

model = applications.VGG16(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3))
for layer in model.layers[:5]:
    layer.trainable = False
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(2, activation="softmax")(x)
model_final = Model(inputs=model.input, output=predictions)

model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
                    metrics=["accuracy"])
train_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True, fill_mode="nearest", zoom_range=0.3,
                                   width_shift_range=0.3, height_shift_range=0.3, rotation_range=30)
test_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True, fill_mode="nearest", zoom_range=0.3,
                                  width_shift_range=0.3, height_shift_range=0.3, rotation_range=30)

train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_height, img_width),
                                                    batch_size=batch_size, class_mode="categorical")
validation_generator = test_datagen.flow_from_directory(validation_data_dir, target_size=(img_height, img_width),
                                                        class_mode="categorical")
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False,
                             mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

model_final.fit_generator(train_generator, samples_per_epoch=nb_train_samples, nb_epoch=epochs,
                          validation_data=validation_generator, nb_val_samples=nb_validation_samples,
                          callbacks=[checkpoint, early])

im = cv2.resize(cv2.imread('data/test/gaff2.jpg'), (img_width, img_height))
im = np.expand_dims(im, axis=0).astype(np.float32)
im = preprocess_input(im)
out = model_final.predict(im)
print(out)
print(np.argmax(out))
