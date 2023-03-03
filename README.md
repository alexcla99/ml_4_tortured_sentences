# Machine Learning for tortured sentences exctraction

In this repository, we propose a first machine learning approach to extract tortured sentences from suspicious scientific literature.

This repository is separated in two directories:
- The `dataset_analysis` provides the used Python code to build and explore a dataset from scratch.
- The `supervised_learning` provides the used Python code to train machine learning models for tortured sentences extraction.

## Dataset analysis

This directory contains four files:
- The `assessments.csv` which contains the suspicious literature metadata such as DOI and manually extracted tortured sentences.
- The `dataset.ipynb` IPython notebook exlaining the whole process to build and explore the dataset used for tortured sentences extraction.
- The `dataset_metadata.csv` which contains the metadata (DOI and tortured sentences) of the dataset used for tortured sentences extraction.
- The `dependencies.txt` listing the used Python version and associated libraries.

## Supervised learning

This directory contains eight files:
- The `cnn.zip` archive containing the CNN model with its best weights and its results for tortured sentences extraction.
- The `freeze.txt` listing the used Python version and associated libraries.
- The `lstm.zip` archive containing the LSTM (RNN) model with its best weights and its results for tortured sentences extraction.
- The `models.py` containing the models implementations.
- The `rcnn.zip` archive containing the RCNN model with its best weights and its results for tortured sentences extraction.
- The `run_train.sh` shell script to run the models training.
- The `supervised_learning.ipynb` IPython notebook explaining the whole process to build the labilsed dataset and display the obtained results.
- The `train_models.py` containing the code to train the implemented models.

## More information

The papers contents used in our dataset cannot be accessible through this repository since some of them are not freely accessible.

We thank Guillaume Cabanc, Cyril Labbé and Alexander Magazinov for their precious work on the depollution of the scientific litterature through the [Problematic Paper Screener](https://www.irit.fr/~Guillaume.Cabanac/problematic-paper-screener).
