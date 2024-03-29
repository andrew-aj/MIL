import tensorflow as tf


def createConvLayer(dropout_rate, input, filters, kernel_size=2, strides=2):
    layer = tf.keras.layers.Conv2D(filters, kernel_size, strides)(input)
    layer = tf.keras.layers.Dropout(dropout_rate)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.ReLU()(layer)
    return layer


def createDenseLayer(dropout_rate, input, nodes, activationLayer=True):
    if activationLayer:
        layer = tf.keras.layers.Dense(nodes, activation='relu')(input)
    else:
        layer = tf.keras.layers.Dense(nodes)(input)
    layer = tf.keras.layers.Dropout(dropout_rate)(layer)
    return layer


def createModel(dropout_rate=0.15):
    # The input for the color map
    colorInput = tf.keras.layers.Input(shape=(None, None, 3), name="colorInput")
    # The input for the depth map. Should be the same size.
    coordinateInput = tf.keras.layers.Input(shape=(None, None, 3), name="coordinateInput")
    # The input for the 3D model of the object.
    modelInput = tf.keras.layers.Input(shape=(None, None, 3), name="modelInput")

    colorLayers = createConvLayer(dropout_rate, colorInput, 32)
    colorLayers = createConvLayer(dropout_rate, colorLayers, 64)
    colorLayers = createConvLayer(dropout_rate, colorLayers, 128)
    colorLayers = createConvLayer(dropout_rate, colorLayers, 256)

    coordinateLayers = createConvLayer(dropout_rate, coordinateInput, 32)
    coordinateLayers = createConvLayer(dropout_rate, coordinateLayers, 64)
    coordinateLayers = createConvLayer(dropout_rate, coordinateLayers, 128)
    coordinateLayers = createConvLayer(dropout_rate, coordinateLayers, 256)

    modelLayers = createConvLayer(dropout_rate, modelInput, 32)
    modelLayers = createConvLayer(dropout_rate, modelLayers, 64)
    modelLayers = createConvLayer(dropout_rate, modelLayers, 128)
    modelLayers = createConvLayer(dropout_rate, modelLayers, 256)

    combinedLayers = tf.keras.layers.concatenate([tf.keras.layers.GlobalMaxPooling2D()(colorLayers),
                                                  tf.keras.layers.GlobalMaxPooling2D()(coordinateLayers)])
    combinedLayers = createDenseLayer(dropout_rate, combinedLayers, 512)
    combinedLayers = createDenseLayer(dropout_rate, combinedLayers, 256)
    combinedLayers = createDenseLayer(dropout_rate, combinedLayers, 128)

    combinedLayers = tf.keras.layers.concatenate([combinedLayers, tf.keras.layers.GlobalMaxPooling2D()(modelLayers)])
    combinedLayers = createDenseLayer(dropout_rate, combinedLayers, 512)
    combinedLayers = createDenseLayer(dropout_rate, combinedLayers, 256)
    combinedLayers = createDenseLayer(dropout_rate, combinedLayers, 128)


    combinedLayers = createDenseLayer(dropout_rate, combinedLayers, 64, False)
    combinedLayers = createDenseLayer(dropout_rate, combinedLayers, 32, False)
    combinedLayers = createDenseLayer(dropout_rate, combinedLayers, 16, False)

    predictions = tf.keras.layers.Dense(6)(combinedLayers)

    model = tf.keras.Model(inputs=[colorInput, coordinateInput, modelInput], outputs=predictions)
    return model


model = createModel()
model.compile(optimizer=tf.keras.optimizers.Adadelta(1), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False))
print("done")

#uncommet when input and data is avilable
#model.fit({"colorInput": colorInput, "coordinateInput": coordinateInput, "modelInput": modelInput}, coordinates, epochs=10, batch_size=32)

#https://www.tensorflow.org/guide/keras/functional
#https://towardsdatascience.com/implementing-a-fully-convolutional-network-fcn-in-tensorflow-2-3c46fb61de3b