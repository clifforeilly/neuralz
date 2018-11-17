from grpc.beta import implementations
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from PIL import Image
import numpy as np


tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS


def main(_):

    host, port = FLAGS.server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    # Change test image to RGB mode
    img = Image.open(FLAGS.image)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Optional: Resize image to 256 x 256. My model was trained on 256 x 256 images
    width_height = (256, 256)
    img = img.resize(width_height)

    # Reshape the image data
    image_data = np.asarray(img, dtype=np.float32)
    image_data = np.expand_dims(image_data, axis=0)
    image_data.reshape((1,) + image_data.shape)

    # Scale down image. New range is 0-1
    image_data = image_data / 255.

    # Create PredictRequest ProtoBuf from image data
    request = predict_pb2.PredictRequest()
    request.model_spec.name = "img"
    request.model_spec.signature_name = "predict"
    request.inputs["images"].CopyFrom(
        tf.contrib.util.make_tensor_proto(image_data, dtype="float32", shape=[1, 256, 256, 3]))

    # Call the TFServing Predict API
    result = stub.Predict(request, 10.0)
    print(result)


if __name__ == '__main__':
    tf.app.run()
