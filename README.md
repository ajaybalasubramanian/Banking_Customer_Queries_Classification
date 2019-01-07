# Consumer_Complain_classifier


[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Objective of this Software code is to perform data classification on Consumer complain data available on https://www.consumerfinance.gov/data-research/consumer-complaints/ . There are two main classifer here

  - Product classifier
  - Issue classifer 

Product is the type of products the consumer identified in the complaint
Issue is the type of issues the consumer identified in the complaint

# Data 
Consumer complaints are added to this public database after the company has responded to the complaint, confirming a commercial relationship with the consumer, or after they've had the complaint for 15 calendar days, whichever comes first. It does’t verify all the facts alleged in complaints, but do give companies the opportunity to publicly respond to complaints by selecting responses from a pre-populated list. 

Database contains 1,192,904 total complaints with 18+ products and 61+ Issues. For this prototype we have considered 6 products each with 500 issues and total 36 issues. 

Data can be downloaded from - https://www.consumerfinance.gov/data-research/consumer-complaints/search/?from=0&searchField=all&searchText=&size=25&sort=created_date_desc

### Tech

This Software usage few Open source libraries to build entire package:

* [NLTK] - The Natural Language Toolkit, or more commonly NLTK, is a suite of libraries and programs for symbolic and statistical natural language processing (NLP) for English written in the Python programming language.NLTK is intended to support research and teaching in NLP or closely related areas, including empirical linguistics, cognitive science, artificial intelligence, information retrieval, and machine learning.
* [Scikit Learn] - It features various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy.
* [Numpy]- NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays. 
* [Pickle] - The pickle module implements binary protocols for serializing and de-serializing a Python object structure. “Pickling” is the process whereby a Python object hierarchy is converted into a byte stream, and “unpickling” is the inverse operation, whereby a byte stream (from a binary file or bytes-like object) is converted back into an object hierarchy. Pickling (and unpickling) is alternatively known as “serialization”, “marshalling,” or “flattening”; however, to avoid confusion, the terms used here are “pickling” and “unpickling”.
* [Python] - Python is an interpreted, high-level, general-purpose programming language. 

### Installation

Dillinger requires nltk, scikit learn, pickle, numpy, scipy, python 2.7

Install the dependencies and devDependencies and start the server.

1. NLTK
```sh
$ sudo pip install -U nltk
```
2. scikit learn
```sh
$ pip install sklearn
```
3. Numpy
```sh
$ sudo pip install -U numpy
```
4. Scipy
```sh
$ pip install scipy
```
5. Pickle
```sh
$ pip install pickle
```


### How to install the software 

Installing software requred installing dependent libraries. Please check Tech section, I have listed required library along with their installation command. Apart from mentioned libraries there may require few common python packages if they are not installed already

### How to use the software
This software has been trained with complain dataset. So you dont require to create data and retrain it. If you stil want change data and retrain, you can reffer ipython notebook, where you can read downlaoded csv or json file and create training data folders as per whichever products/issues you select for training.

This software contains two different model
1. Product model
2. Issue model

You can run each of this model separately and test it. 
```
python svm_classifier_product.py
```
Above command will give you score and product name for identified product for given complain

```
python svm_classifier_issues.py
```
Above command will give you score and issue name for identified issue for given complain

Once you train it, it will store model in pickle file. For all subsequent try it will load model from pickle so will not consume time in model building.

Now once you have tained both model, you can test both model together using svn_classifer_mix.py file. 
```
python svm_classifier_mix.py
```

This command will provide you score for identify product and issue both for given complain.

When you run any of this model, it will read trainig dataset, build model and store it in pickle file. After that it will ask you to enter the complain text - 

