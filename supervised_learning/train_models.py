# Imports
from tensorflow.config.threading import set_inter_op_parallelism_threads, set_intra_op_parallelism_threads, get_inter_op_parallelism_threads, get_intra_op_parallelism_threads
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import pandas as pd
import numpy as np
import io, os, pickle, sys
import models as m

# Python 3.6 imports
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras as K

# Python 3.10 imports
#from keras.utils import pad_sequences
#import keras as K

## TODO use Keras-Tuner for random hyperparameters search
## TODO use LR reducer
## TODO use L1 / L2 regularisers
## TODO use val_roc_auc and max mode for monitoring
## TODO use tuned optimisers parameters

# Loading arguments
if len(sys.argv) < 2:
	sys.exit("Usage: python3 train_models.py <model_name>")
MODEL = sys.argv[1]

# Multothreading configuration
NUM_THREADS = 24
set_inter_op_parallelism_threads(NUM_THREADS)
set_intra_op_parallelism_threads(NUM_THREADS)
print("Using %d threads for parallelism between independant operations" % (get_inter_op_parallelism_threads()))
print("Using %d threads for parallelism between individual operations" % (get_intra_op_parallelism_threads()))

# Data parameters
SEED = 1337
DATASET = "labelised_dataset.csv"
TEST_SIZE = .15
VAL_SPLIT = .2

# Training parameters
LOSS = "binary_crossentropy"
METRICS = [
    K.metrics.BinaryAccuracy(),
    K.metrics.AUC(curve="ROC", name="roc_auc"),
    K.metrics.AUC(curve="PR", name="pr_auc"),
    K.metrics.TruePositives(),
    K.metrics.FalsePositives(),
    K.metrics.TrueNegatives(),
    K.metrics.FalseNegatives()
]
BATCH_SIZE = 128
EPOCHS = 200
ES_DELTA = 1e-4
ES_PATIENCE = 10
MONITOR = "val_loss"
MODE = "auto"

# Loading the dataset
content = pd.read_csv(DATASET)
content["sentence"] = content["sentence"].apply(str)
content["class"] = content["class"].apply(float)
X_content = content["sentence"]
Y_content = content["class"]

# Using the raw dataset
def use_raw_dataset(X:pd.Series, Y:pd.Series) -> tuple:
	return X, Y

# Undersampling the dataset
def undersample_dataset(
	X:pd.Series,
	Y:pd.Series,
	st:float=1.
) -> tuple:
	undersample = RandomUnderSampler(
		random_state=SEED,
		sampling_strategy=st
	)
	X, Y = undersample.fit_resample(X, Y)
	return X, Y

# Applying SMOTE on the dataset
def smote_on_dataset(
	X:pd.Series,
	Y:pd.Series,
	st:float=.2
) -> tuple:
	oversample = SMOTE(
		random_state=SEED,
		sampling_strategy=st
	)
	X, Y = oversample.fit_resample(X, Y)
	return X, Y

# Oversampling the dataset
def oversample_dataset(
	X:pd.Series,
	Y:pd.Series,
	st:float=1.
) -> tuple:
	oversample = RandomOverSampler(
		random_state=SEED,
		sampling_strategy=st
	)
	X, Y = oversample.fit_resample(X, Y)
	return X, Y

# Function to compute MCC
def compute_mcc(accr:list) -> float: #|None:
    tp = accr[4]
    fp = accr[5]
    tn = accr[6]
    fn = accr[7]
    denom = np.sqrt([(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)])
    if denom == 0:
        return None
    mcc = ((tp*tn)-(fp*fn)) / denom
    return mcc[0]

# Strategies
strategies = {
	"raw": use_raw_dataset,              # OK
	"undersampled": undersample_dataset, # OK
	"smote": smote_on_dataset,           # OK
	"oversampled": oversample_dataset    # OK
}

# Models
models = {
	"lstm": [m.lstm, "rmsprop"], # OK
	"cnn": [m.cnn, "adam"],      # OK
	"rcnn": [m.rcnn, "adam"],    # OK
	#"rmdl": [m.rmdl, ""]         # NOT YET IMPLEMENTED
}
if MODEL not in models.keys():
	sys.exit("Bad model name (%s given, available models are: %s)" % (MODEL, str(", ".join(models.keys()))))

# Displaying the used model
print("==================== %s ====================" % (MODEL))

# Training the model following these strategies
for k, v in strategies.items():

	# Displaying the current strategy
	print("==================== %s ====================" % (k))

	# Building the associated dataset
	tok = Tokenizer()
	tok.fit_on_texts(X_content)
	X = tok.texts_to_sequences(X_content)
	for i in range(len(X)):
		if len(X[i]) < 2:
			X[i].append(0)
	X, Y = v(X, Y_content)
	print("X length: %d" % (len(X)))
	print("Y length: %d" % (len(Y)))

	# Stratifying the dataset and weighting classes when using raw data and SMOTE
	stratify = None
	weights = None
	if k in ["raw", "smote"]:
		stratify = Y
		class_weights = list(class_weight.compute_class_weight(
		    "balanced",
		    classes=np.unique(Y),
		    y=Y
		))
		weights = dict()
		for index, weight in enumerate(class_weights):
			weights[index] = weight

	# Building the subsets
	X_train, X_test, Y_train, Y_test = train_test_split(
		X,
		Y,
		test_size=TEST_SIZE,
		random_state=SEED,
		stratify=stratify
	)
	print("X_train length: %d" % (len(X_train)))
	print("Y_train length: %d" % (len(Y_train)))
	print("X_test length: %d" % (len(X_test)))
	print("Y_test length: %d" % (len(Y_test)))

	# Uniformising sentences
	seq_mat = pad_sequences(X_train)
	print("X_train sequence matrix shape: %s" % (
		str(seq_mat.shape)
	))

	# Defining callbacks
	best_model = os.path.join(
	        	MODEL,
	        	k,
	        	("%s_%s_best_model.tf" % (MODEL, k))
	        )
	callbacks = [
	    K.callbacks.EarlyStopping(
	        monitor=MONITOR,
	        min_delta=ES_DELTA,
	        patience=ES_PATIENCE,
			mode=MODE
	    ),
	    K.callbacks.ModelCheckpoint(
	        monitor=MONITOR,
	        filepath=best_model,
	        save_best_only=True,
			mode=MODE
	    )
	]

	# Defining the model
	model = models[MODEL][0](
			input_shape=seq_mat.shape[-1],
			vocab_size=np.max(seq_mat)+1,
			name=k,
			seed=SEED
		)

	# Compiling the model
	model.summary()
	model.compile(
	    loss=LOSS,
	    optimizer=models[MODEL][1],
	    metrics=METRICS
	)

	# Fitting the model
	train_metrics = model.fit(
	    seq_mat,
	    Y_train,
	    class_weight=weights,
	    batch_size=BATCH_SIZE,
	    epochs=EPOCHS,
	    validation_split=VAL_SPLIT,
	    callbacks=callbacks
	)

	# Saving the train metrics
	with io.open(os.path.join(
		MODEL,
		k,
		("%s_%s_train_metrics.pickle" % (MODEL, k))
	), "wb") as handle:
		pickle.dump(train_metrics.history, handle, pickle.HIGHEST_PROTOCOL)
		handle.close()

	# Testing the model
	seq_mat = pad_sequences(X_test)
	print("X_test sequence matrix shape: %s" % (str(seq_mat.shape)))
	model = K.models.load_model(best_model)
	test_metrics = model.evaluate(seq_mat, Y_test)
	mcc = compute_mcc(test_metrics)
	print("Test MCC is %s" % (str(mcc)))
	test_metrics.append(mcc)

	# Saving the test metrics
	with io.open(os.path.join(
		MODEL,
		k,
		("%s_%s_test_metrics.pickle" % (MODEL, k))
	), "wb") as handle:
		pickle.dump(test_metrics, handle, pickle.HIGHEST_PROTOCOL)
		handle.close()

# Finished
print("====================\nDone.")
