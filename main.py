import tensorflow as tf


def createConvLayer(dropout_rate, input, filters, kernel_size=2, strides=2):
    layer = tf.keras.layers.Conv2D(filters, kernel_size, strides)(input)
    layer = tf.keras.layers.Dropout(dropout_rate)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.MaxPool2D()(layer)
    layer = tf.keras.layers.ReLU()(layer)
    return layer


def createDenseLayer(dropout_rate, input, nodes):
    layer = tf.keras.layers.Dense(nodes, activation='relu')(input)
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

    combinedLayers = tf.keras.layers.concatenate([colorLayers, coordinateLayers])
    combinedLayers = createConvLayer(dropout_rate, combinedLayers, 32, 1, 1)
    combinedLayers = createConvLayer(dropout_rate, combinedLayers, 64, 1, 1)
    combinedLayers = createConvLayer(dropout_rate, combinedLayers, 128, 1, 1)
    combinedLayers = createConvLayer(dropout_rate, combinedLayers, 256, 1, 1)

    combinedLayers = tf.keras.layers.concatenate([combinedLayers, modelLayers])
    combinedLayers = createConvLayer(dropout_rate, combinedLayers, 32, 1, 1)
    combinedLayers = createConvLayer(dropout_rate, combinedLayers, 64, 1, 1)
    combinedLayers = createConvLayer(dropout_rate, combinedLayers, 128, 1, 1)
    combinedLayers = createConvLayer(dropout_rate, combinedLayers, 256, 1, 1)

    combinedLayers = tf.keras.layers.GlobalMaxPooling2D()(combinedLayers)
    combinedLayers = tf.keras.layers.Flatten()(combinedLayers)
    combinedLayers = createDenseLayer(dropout_rate, combinedLayers, 128)
    combinedLayers = createDenseLayer(dropout_rate, combinedLayers, 256)
    combinedLayers = createDenseLayer(dropout_rate, combinedLayers, 512)

    predictions = tf.keras.layers.Dense(6)(combinedLayers)

    model = tf.keras.Model(inputs=[colorInput, coordinateInput, modelInput], outputs=predictions)
    return model


model = createModel()
model.compile(optimizer=tf.keras.optimizers.Adadelta(1), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False))

#uncommet when input and data is avilable
#model.fit({"colorInput": colorInput, "coordinateInput": coordinateInput, "modelInput": modelInput}, coordinates, epochs=10, batch_size=32)