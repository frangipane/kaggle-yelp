# Kaggle - Yelp Restaurant Photo Classification

Contest details: https://www.kaggle.com/c/yelp-restaurant-photo-classification

Thanks to kaggle competitor Nina Chen for providing
[starter code](https://github.com/ncchen55414/Kaggle-Yelp/tree/master/CNN_Submission1)
for this competition.

## Explore raw data
`0_explore_data.ipynb` (uses `R`, not python)

## Extract features using from pre-trained convolutional neural network
`1_feature-extraction-bvlc_reference_caffenet.ipynb`
- use [Caffe](http://caffe.berkeleyvision.org/) framework
- pre-trained
  [BVLC CaffeNet model](https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet),
  based on Alexnet
- extracts features from fc6, fc7, and softmax (fully-connected) layers
  
## Remove path names from photo ids
`2_process_h5_files.ipynb`
- better to remove this step entirely by changing the way files are
  saved in the previous step

## Visualize extracted features
`3_feature_visualization.ipynb`
- visualize accuracy of predicted class (from 1000-dim softmax probability
  vectors), though these vectors aren't used in the model training
- make t-SNE plots of fc6, fc7, and concatenated fc6+fc7 features,
  grouping by label
- multi-instance problem - average the fc6 and fc7 photo feature
  vectors by restaurant

## Train SVM on extracted features
`4_classification.ipynb`
- concatenate fc6 and fc7 layer features
- train SVM model with rbf kernel (seems to perform better than linear SVM)
- multi-label problem - take simplest approach, a.k.a. "binary
  relevance", by training each label separately
- coarser-grained grid validation was performed, but omitted in this
  notebook (however, finer-grained validation results remain).
  Optimize parameters by validation on 20% of training set (more
  time-consuming cross-validation or optimization per label was not
  performed)
