%
% The SVM.1 approach in the paper:
%
% David D. Lewis, Yiming Yang, Tony G. Rose, and Fan Li.
% RCV1: A new benchmark collection for text categorization research.
% Journal of Machine Learning Research, 5:361-397, 2004.
%
% We use liblinear as the classifier. 
% The classification algorithm can be either l2-loss, l1-loss SVM or logistic regression


clear all;
close all;
clc;

addpath(genpath('/home/chchengl/Greek/LIBSVM/'));

% algo = 'l2svm_dual';
% algo = 'lr';
algo = 'l1svm_dual';

nr_fold = 3;


save_csv_file_name = strcat(['/home/chchengl/Greek/Submission/LIBSVM/LIBSVM_prediction_',num2str(nr_fold),'fold_[',algo,'].csv']);
save_model_file_name = strcat([save_csv_file_name(1:end-3),'mat']);


disp('INFO: loading training data')
[y_train, X_train, label_map] = read_sparse_ml('/home/chchengl/Greek/Data/wise2014-train.libsvm');
y_train = full(y_train);
y_train = 2*y_train - 1;
disp('INFO: loading testing data')
[~, X_test, ~] = read_sparse_ml('/home/chchengl/Greek/Data/wise2014-test.libsvm');


% some useful parameters
numTrain = size(X_train, 1);
numTest = size(X_test, 1);
numFeat = size(X_train, 2);
numLabel = size(y_train, 2);
numLabel = 10;

y_train = y_train(:, 1:numLabel);

% add bias
bias = 1;
X_train = [bias*ones(numTrain, 1), X_train];
X_test = [bias*ones(numTest, 1), X_test];


% initialize (w, b) matrices for all labels
W = zeros(size(X_train, 2), numLabel);
B = zeros(1, numLabel);
% obtain (w,b) for each label
rand('seed', 2014);
matlabpool open 12
parfor i = 1:numLabel
    fprintf(1, 'INFO: training label %d (i %d)...\n', label_map(i), i);
    [W(:, i), B(i)] = train_one_label(X_train, y_train(:, i), algo, nr_fold);
end
matlabpool close

pred = 2*(bsxfun(@plus, X_train*W, B)>0) - 1;
F = mean_Fscore(y_train, pred);
disp(F);

% make prediction
pred = bsxfun(@plus, X_test*W, B);
pred = (pred>0);
% make submission
make_submission(pred, label_map, save_csv_file_name);
% save model in mat format
% save(save_model_file_name, 'W', 'B', 'pred');