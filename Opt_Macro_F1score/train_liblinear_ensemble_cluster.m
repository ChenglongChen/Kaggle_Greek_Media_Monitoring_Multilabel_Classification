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
nr_round = 100;
feature_ratio = 1.0;

save_csv_file_name = strcat(['/home/chchengl/Greek/Submission/LIBSVM/LIBSVM_ensemble_[Feature',num2str(feature_ratio),']_',...
    '[Round',num2str(nr_round),']_',...
    '[Fold',num2str(nr_fold),']_',...
    '[',algo,']','.csv']);
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
% numLabel = 10;

pred = zeros(numTest, numLabel, nr_round);
rand('seed', 2014);
matlabpool open 12
for round = 1:nr_round
    %sample_index = randi(numTrain, numTrain, 1);
    sample_index = randperm(numTrain, numTrain);
%     sample_index = 1:numTrain;
    feature_index = randperm(numFeat, floor(numFeat*feature_ratio));
%     feature_index = 1:numFeat;
    % add bias
    bias = 1;
    X_train2 = [bias*ones(numTrain, 1), X_train(sample_index, feature_index)];
    y_train2 = y_train(sample_index, :);
    X_test2 = [bias*ones(numTest, 1), X_test(:, feature_index)];
    
    % initialize (w, b) matrices for all labels
    W = zeros(size(X_train2, 2), numLabel);
    B = zeros(1, numLabel);
    % obtain (w,b) for each label    
    parfor i = 1:numLabel
        %fprintf(1, 'INFO: training label %d (i %d)...\n', label_map(i), i);
        [W(:, i), B(i)] = train_one_label(X_train2, y_train2(:, i), algo, nr_fold);
    end
    % make prediction
    pred(:, :, round) = bsxfun(@plus, X_test2*W, B);
end
matlabpool close

% make binary prediction
pred1 = (median(pred, 3)>0);
pred2 = (mean((pred>0),3)>0.5);
% make submission
save_csv_file_name_rule1 = strcat([save_csv_file_name(1:end-4),'_rule1.csv']);
make_submission(pred1, label_map, save_csv_file_name_rule1);
save_csv_file_name_rule2 = strcat([save_csv_file_name(1:end-4),'_rule2.csv']);
make_submission(pred2, label_map, save_csv_file_name_rule2);
% % save model in mat format
% save(save_model_file_name, 'W', 'B', 'pred', 'pred1', 'pred2');