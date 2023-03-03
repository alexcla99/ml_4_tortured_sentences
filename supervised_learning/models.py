# Python 3.6 imports
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras as K

# Python 3.10 imports
#from keras.utils import pad_sequences
#import keras as K

# Layers parameters
EMB_OUT_DIM = 50
LSTM_UNITS = 64
CONV_FILTERS = 64
FC1_UNITS = 256

# LSTM
def lstm(input_shape:int, vocab_size:int, name:str, seed:int) -> K.Model:
    return K.Sequential(
		[
			K.layers.Input(
				shape=[input_shape],
				name="%s_inputs" % (name)
			),
			K.layers.Embedding(
				input_dim=vocab_size,
				output_dim=EMB_OUT_DIM,
				input_length=input_shape,
				name="%s_embedding" % (name)
			),
			K.layers.LSTM(
				units=LSTM_UNITS,
				name="%s_lstm" % (name)
			),
			K.layers.Dropout(
				rate=.5,
				seed=seed,
				name="%s_dropout" % (name)
			),
			K.layers.Dense(
				units=FC1_UNITS,
				name="%s_fc1" % (name)
			),
			K.layers.Activation(
				activation="relu",
				name="%s_relu" % (name)
			),
			K.layers.Dense(
				units=1,
				name="%s_fc2" % (name)
			),
			K.layers.Activation(
				activation="sigmoid",
				name="%s_sigmoid" % (name)
			)
		],
		name="%s_lstm" % (name)
	)

# CNN
def cnn(input_shape:int, vocab_size:int, name:str, seed:int) -> K.Model:
	return K.Sequential(
		[
			K.layers.Input(
				shape=[input_shape],
				name="%s_inputs" % (name)
			),
			K.layers.Embedding(
				input_dim=vocab_size,
				output_dim=EMB_OUT_DIM,
				input_length=input_shape,
				name="%s_embedding" % (name)
			),
			K.layers.Conv1D(
				filters=CONV_FILTERS,
				kernel_size=2,
				activation="relu",
				name="%s_conv" % (name)
			),
			K.layers.Dropout(
				rate=.5,
				seed=seed,
				name="%s_dropout" % (name)
			),
			K.layers.MaxPooling1D(
				pool_size=1,
				name="%s_maxpooling" % (name)
			),
			K.layers.Flatten(
				name="%s_flatten" % (name)
			),
			K.layers.Dense(
				units=FC1_UNITS,
				name="%s_fc1" % (name)
			),
			K.layers.Activation(
				activation="relu",
				name="%s_relu" % (name)
			),
			K.layers.Dense(
				units=1,
				name="%s_fc2" % (name)
			),
			K.layers.Activation(
				activation="sigmoid",
				name="%s_sigmoid" % (name)
			)
		],
		name="%s_cnn" % (name)
	)

# RCNN
def rcnn(input_shape:int, vocab_size:int, name:str, seed:int) -> K.Model:
    return K.Sequential([
		K.layers.Input(
				shape=[input_shape],
				name="%s_inputs" % (name)
			),
			K.layers.Embedding(
				input_dim=vocab_size,
				output_dim=EMB_OUT_DIM,
				input_length=input_shape,
				name="%s_embedding" % (name)
			),
			K.layers.Conv1D(
				filters=CONV_FILTERS,
				kernel_size=2,
				activation="relu",
				name="%s_conv" % (name)
			),
			K.layers.Dropout(
				rate=.25,
				seed=seed,
				name="%s_dropout" % (name)
			),
			K.layers.MaxPooling1D(
				pool_size=1,
				name="%s_maxpooling" % (name)
			),
			K.layers.LSTM(
				units=LSTM_UNITS,
				recurrent_dropout=.2,
				name="%s_lstm" % (name)
			),
			K.layers.Dense(
				units=FC1_UNITS,
				name="%s_fc1" % (name)
			),
			K.layers.Activation(
				activation="relu",
				name="%s_relu" % (name)
			),
			K.layers.Dense(
				units=1,
				name="%s_fc2" % (name)
			),
			K.layers.Activation(
				activation="sigmoid",
				name="%s_sigmoid" % (name)
			)
		],
		name="%s_rcnn" % (name)
	)

# RMDL
#def rmdl(input_shape:int, vocab_size:int, name:str, seed:int) -> K.Model:
#    return K.Sequential([
#
#	])