import numpy as np
from tensorflow import keras

array_2D = np.zeros((200,300))

def super_resolve(array_2D):
    # Change model input shape to accept all size inputs
    model = tf.keras.models.load_model('C://Users//Tara//Desktop//generator_x2.h5', compile=False)
    inputs = tf.keras.Input((None, None, 3))
    output = model(inputs)
    model = keras.models.Model(inputs, output)
        
    # Convert 2D_array to 3-channel np.array
    low_res = np.asarray(array_2D)
    low_res = np.repeat(low_res, 3, axis=2)

    # Rescale to 0-1.
    low_res = low_res / 255.0

    # Get super resolution image
    sr = model.predict(np.expand_dims(low_res, axis=0))[0]

    # Rescale values in range 0-255
    sr = ((sr + 1) / 2.) * 255

    # Return to 1-channel
    sr_2D = sr[0]

    return sr_2D

sr = super_resolve(array_2D)
print(sr)