# Machine Learning for tortured sentences exctraction

Author: Alexandre Clausse.

## Summary of the work presented in this repository

In a context where more and more scientific papers are partially or totally generated and plagiarised [1], some of these regular-looking papers actually contain tortured sentences [2] (e.g. "profund learning" which is used in the psychologic field by which should be "deep learning" in the field). In this repository, we propose a first machine learning approach to extract such sentences from suspicious scientific literature, based on the [Problematic Paper Screener](https://www.irit.fr/~Guillaume.Cabanac/problematic-paper-screener) [3] (PPS) "Tortured" detector assessments.

This repository is separated in two parts, the *dataset_analysis* directory provides the used code to build and explore a dataset from scratch, and the *supervised_learning* provides the used code to train several machine learning models to automatically extract tortured sentences.

The *dataset_analysis* directory contains four files:
- The *assessments.csv* file which contains the suspicious literature metadata such as DOI and manually extracted tortured sentences, gathered from the PPS.
- The *dataset.ipynb* IPython notebook exlaining the whole process to build and explore the dataset used for tortured sentences extraction.
- The *dataset_metadata.csv* file which contains the metadata (DOI and tortured sentences) of the dataset used for tortured sentences extraction.
- The *dependencies.txt* file listing the used Python version and associated libraries.

The *supervised_learning* directory contains eight files:
- The *cnn.zip* archive containing the CNN model with its best weights and its results for tortured sentences extraction.
- The *freeze.txt* file listing the used Python version and associated libraries.
- The *lstm.zip* archive containing the LSTM (RNN) model with its best weights and its results for tortured sentences extraction.
- The *models.py* script containing the models implementations with Python.
- The *rcnn.zip* archive containing the RCNN model with its best weights and its results for tortured sentences extraction.
- The *run_train.sh* Shell script to run the models training.
- The *supervised_learning.ipynb* IPython notebook explaining the whole process to build the labilsed dataset and display the obtained results.
- The *train_models.py* script containing the Python code to train the implemented models.

**Remark: the papers contents used in our dataset cannot be accessible through this repository since some of them are not freely available.**

## Acknowledgements and references

We thank Guillaume Cabanc, Cyril Labbé and Alexander Magazinov for their precious work on the depollution of the scientific litterature through the PPS.

[1] Cyril Labbé and Dominique Labbé. Duplicate and fake publications in the scientific literature: how many scigen papers in the computer science? *Scientometrics*, June 2012.

[2] Guillaume Cabanac and Cyril Labbé. Prevalence of nonsensical algorithmically generated papers in the scientific litearture. *Journal of the Association for Information Science and Technology*, 72(12):1461-1476, 2021.

[3] Guillaume Cabanac, Cyril Labbé, and Alexander Magazinov. The 'problematic paper screener' automatically selects suspect publications for postpublication (re)assessment, 2022.
