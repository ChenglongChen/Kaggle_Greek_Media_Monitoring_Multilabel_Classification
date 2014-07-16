# Kaggle's Greek Media Monitoring Multilabel Classification (WISE 2014)
  
This repo holds the MATLAB code I used to make submision to [Kaggle's Greek Media Monitoring Multilabel Classification (WISE 2014)](http://www.kaggle.com/c/wise-2014). The score using this implementation is **0.75342**, ranking 24th out of 121 teams. (That entry is placed in `./Submission` folder.)

For the MATLAB code, there is also a published html page in the `./Opt_Mean_F1score/html` folder.


## Method

It implements the following methods to tackle the multilabel classification problem:

* **Binary Relevance** (BR) and **Classifier Chains** (CC) methods to transform the multi-label classification problem into binary one [1].

* Linear SVM (from LIBLINEAR package) as the "base classifier". In specific, I implemented the SVM.1 and SCutFBR.1 approach as described in [2] and [3].


## Requirement

- [LIBLINEAR](http://www.csie.ntu.edu.tw/~cjlin/liblinear/) For instructions of using LIBLINEAR and additional information, see the README file included in the package and the LIBLINEAR FAQ.^1
  
- [Multi-label classification tool: read_sparse_ml.c](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/multilabel/)^2
  

^1 ^2 For your convenience, I have already included the necessary pre-complied files of LIBLINEAR and `read_sparse_ml.c` in the folder `./utils/LIBLINEAR`. Please see there for copyright information.
  
  
## Instruction

* download data from the [competition website](http://www.kaggle.com/c/wise-2014/data) and put all the data into `./Data` dir:
 - `./Data/wise2014-train.libsvm`
 - `./Data/wise2014-test.libsvm`
 
* put all the code into `./MATLAB` dir:^3
 - `./MATLAB/Opt_Mean_F1score/...`
 - `./MATLAB/utils/...`

* run `train_WISE.m` to create csv submission to Kaggle.

^3 If you don't want to do this, you have to specify the correct path to the data and to the utils in the function `train_WISE.m` (see the code in the beginning).


## Reference
[1] Jesse Read, Bernhard Pfahringer, Geoff Holmes, and Eibe Frank, "Classifier chains for multi-label classification."

[2] David D. Lewis, Yiming Yang, Tony G. Rose, and Fan Li, "RCV1: A new benchmark collection for text categorization research." *Journal of Machine Learning Research*, 5:361-397, 2004.

[3] Rong-En Fan and Chih-Jen Lin, "A Study on Threshold Selection for Multi-label Classification."