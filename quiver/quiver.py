import keras.applications as apps
from quiver_engine.server import launch

model = apps.vgg16.VGG16()
launch(model, input_folder="./")
