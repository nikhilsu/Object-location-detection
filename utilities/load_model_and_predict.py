from keras.models import load_model
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import socket


def __tukey_bi_weight_loss(y_true, y_predicted):
    z = y_true - y_predicted
    z_abs = tf.abs(z)
    c = 4.685
    subset_bool = tf.less_equal(z_abs, c)
    subset = tf.cast(subset_bool, z_abs.dtype)
    inv_subset = tf.cast(tf.logical_not(subset_bool), z_abs.dtype)
    c_sq_by_six = c ** 2 / 6
    return (1 - ((1 - ((z / c) ** 2)) ** 3) * subset + inv_subset) * c_sq_by_six


def capture_and_save_image(filename):
    camera = cv2.VideoCapture(1)
    return_value, image = camera.read()
    smaller_image = cv2.resize(image, (150, 150), interpolation=cv2.INTER_AREA)
    cv2.imwrite(filename, smaller_image)
    camera.release()


model_filename = '../data/pumpkin_model.h5'
test_image_path = 'test_image.jpg'

ip = '192.168.10.117'
port = 51002

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.connect((ip, port))

model = load_model(model_filename, custom_objects={'__tukey_bi_weight_loss': __tukey_bi_weight_loss})

while True:
    capture_and_save_image(test_image_path)

    test_x = np.asarray(Image.open(test_image_path).convert('RGB'), dtype='float64') / 255
    test_x_resized = test_x.reshape(1, 150, 150, 3)

    test_y = model.predict(test_x_resized)
    flattened_test_y = test_y.flatten()
    test_y_to_vicon = '{},{},{}'.format(flattened_test_y[0], flattened_test_y[1], flattened_test_y[2])
    print('Sending data: ' + test_y_to_vicon)

    sock.send(test_y_to_vicon.encode())