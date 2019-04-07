from keras import backend as K
import tensorflow as tf
import numpy as np
import sklearn

filepath = 'model/model'

model = tf.keras.models.load_model(
    filepath,
    compile=True
)

inp = model.input                                           # input placeholder
outputs = [layer.output for layer in model.layers]          # all layer outputs
#functors = [K.function([inp], [out]) for out in outputs]    # evaluation functions
functor = K.function([inp, K.learning_phase()], outputs )   # evaluation function

# Testing
test = np.random.randn(28, 28) # Get desired input
#layer_outs = [func([test]) for func in functors]
layer_outs = functor([test, 1.])
print("hi" + layer_outs)

