import tensorflow as tf

modified_RestNet50 = tf.keras.applications.resnet.ResNet50(weights='imagenet',include_top=False,input_shape=([48, 160, 3]))