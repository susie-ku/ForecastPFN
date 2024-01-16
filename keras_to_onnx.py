from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
import tensorflow as tf
import onnx
import os
os.environ['TF_KERAS'] = '1'
# import keras2onnx
import tf2onnx
from benchmark.utils.metrics import smape

onnx_model_name = 'forecastpfn.onnx'

# # # model = load_model('model-resnet50-final.h5')
# model = tf.keras.models.load_model('/media/ssd-3t/kkuvshinova/time_series/ForecastPFN/saved_weights', custom_objects={'smape': smape})

# # # onnx_model = keras2onnx.convert_keras(model, model.name)
# # onnx.save_model(onnx_model, onnx_model_name)

# # input_signature = [tf.TensorSpec([3, 3], tf.float32, name='x')]
# input_signature = (
#     tf.TensorSpec(shape=(None, 100), dtype=tf.float32, name='history'),
#     tf.TensorSpec(shape=(None, 1, 5), dtype=tf.int64, name='target_ts'),
#     tf.TensorSpec(shape=(None,), dtype=tf.int32, name='task'),
#     tf.TensorSpec(shape=(None, 100, 5), dtype=tf.int64, name='ts'),
# )
# # Use from_function for tf functions
# onnx_model, _ = tf2onnx.convert.from_keras(model, opset=13)
# onnx.save(onnx_model, onnx_model_name)

# from onnx2pytorch import ConvertModel

# onnx_model = onnx.load(onnx_model_name)
# pytorch_model = ConvertModel(onnx_model)
# print(pytorch_model)

from onnx2torch import convert

# Or you can load a regular onnx model and pass it to the converter
onnx_model = onnx.load(onnx_model_name)
torch_model_2 = convert(onnx_model)
print(torch_model_2)