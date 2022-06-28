# Recommendation Engine Model Trials

In this repo I try to build ensemble of models to scores products based on product and client features. To work on recommendation alorithms I use <a href="https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations/overview">H&M  kaggle datased</a>. To handle this kind of big data set I use one of the popular big data framework dask and to accelareta data preprocessing part I use gpu with cudf. This repository has three main parts:

* Data preprocessing
* Content & Collabrative models
* Ensemble and Ranking model
  
In the data processing part I use dask and implement some feature engineering and data cleaning methods. In the secodn part to scores the products based on client preferences I use tf-idf, doc2vec as content based models and I use LDa, LSA, NMF and PCA as collabrative models. In the third part I use LGBM as ensemble model to combine and rerank the products based on the product features, client features and scores from the content and collabrative models.

