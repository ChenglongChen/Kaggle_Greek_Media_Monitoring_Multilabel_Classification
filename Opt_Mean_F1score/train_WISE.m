%% My solution to Greek Media Monitoring Multilabel Classification
% 
% <http://www.kaggle.com/c/wise-2014 Greek Media Monitoring Multilabel Classification (WISE 2014)>
%
% *Author:* Chenglong Chen < <https://www.kaggle.com/users/102203/yr yr@Kaggle> >
% 
% *Email:* c.chenglong@gmail.com
% 
% *Description:*
% 
% - This solution transforms the multi-label classification problem into
% binary classification problmen using two different methods: Binary 
% Relevance (BR) and Classifier Chains (CC). See [1] for details. 
% 
% - For the "base classifier", it uses linear SVM. In specific, I implement
% the SVM.1 and SCutFBR.1 approach as described in [2] and [3].
%
% - We use liblinear since there are ~6w instances and ~30w features. The
% classification algorithm can be either l1-loss, l2-loss SVM or logistic
% regression. (Seems l1-loss SVM works best.)
%
% - This implementation is based on the code in the LIBSVM page, i.e.,
% <http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/multilabel/rcv1/rcv1_lineart_col.m rcv1_lineart_col.m>,
% which optimizes the macro-average F-measure.
% Since the evaluation metric for this competition is mean F1-score, we
% accordingly modify the code above and implement the cyclic optimization
% procedure as detailed in [3]. Compared with the original code, we
% observed improvement around 0.01~0.02 by directly optimizing mean
% F1-score.
% 
% *Requirement:*
% 
% - <http://www.csie.ntu.edu.tw/~cjlin/liblinear/ LIBLINEAR>   
%   For instructions of using LIBLINEAR and additional information, see
%   the README file included in the package and the LIBLINEAR FAQ.
% 
% - <http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/multilabel/read_sparse/read_sparse_ml.c
% Multi-label classification tool: read_sparse_ml.c>
%   
% 
% *Reference:*
% 
% [1] Jesse Read, Bernhard Pfahringer, Geoff Holmes, and Eibe Frank,
% "Classifier chains for multi-label classification."
%
% [2] David D. Lewis, Yiming Yang, Tony G. Rose, and Fan Li,
% "RCV1: A new benchmark collection for text categorization research."
% _Journal of Machine Learning Research_, 5:361-397, 2004.
% 
% [3] Rong-En Fan and Chih-Jen Lin, "A Study on Threshold Selection for
% Multi-label Classification."
% 
%% General configuration for MATLAB
clear all;
close all;
clc;

% use gpucluster or not
% just for my own use...
if isunix  % gpucluster is unix platform
    home_path = '/home/chchengl/Greek';
    matlabpool close force local
    matlabpool open 12
else
    home_path = '../..';
    % comment out these if you don't have Parallel Computing Toolbox
    matlabpool close force local
    matlabpool open 2
end

% path to save csv submission
path_to_save_csv = strcat([home_path, '/Submission/MATLAB/Opt_Mean_F1score/']);
if ~exist(path_to_save_csv, 'dir')
    disp('INFO: Creat submission folder');
    mkdir(path_to_save_csv);
end

% add path to toolbox
addpath(genpath(strcat([home_path, '/MATLAB/Opt_Mean_F1score/'])));
addpath(genpath(strcat([home_path, '/MATLAB/utils/'])));


%% General training parameter configuration

% multi-label classification transformation
% method for transforming multi-label classification problem into binary
% classification problem
% transformation = 'BR';
transformation = 'CC'; % CC works better

% feature normalization method
feat_norm_method = 'raw';
% feat_norm_method = 'binary';
% feat_norm_method = 'binary_inst_norm';
% feat_norm_method = 'binary_feat_norm';
% feat_norm_method = 'log';

% random seed to ensure reproducible results
random_seed = 2014;

% number of rounds to train random shuffle instace/label data set,
% which are used to create the final bagging submission
nr_round = 5;

% randomly draw this ratio of samples in each round
instance_ratio = 1.0;
instance_replacement = false;

% randomly draw this ratio of features (without replacement) in each round
feature_ratio = 1.0;


%% SVM.1 and SCutFBR.1 parameter configuration

% cv folds in SVM.1 and SCutFBR.1
nr_fold = 3;

% list of fbr using in SVM.1
fbr_list = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8];
% fbr_list = 0.05:0.05:0.95;

% termination criterion for cyclic optimization in SCutFBR.1
improve_tol = 0.001;

% verbose in SCutFBR.1 
% 0 - quite mode
% 1 - fold level
% 2 - fold & iteration level
% 3 - fold & iteration & label level
verbose = 3;


%% Linear SVM parameter configuration

% which linear SVM to use
% During the competition, I only tried the first four and l2reg_l1loss_dual
% gives the best and is fast
algorithms = { 'l2reg_lr_primal',...
               'l2reg_l2loss_dual',...
               'l2reg_l2loss_primal',...
               'l2reg_l1loss_dual',...
               'crammer_singer_svc',...
               'l1reg_l2loss_svc',...
               'l1reg_lr',...
               'l2reg_lr_dual',...
               };
algorithms = {'l2reg_l1loss_dual'};
algorithms = {'l1reg_l2loss_svc'};

% cost penalty for linear SVM
% try varying C and manually select the best, though it should be based on
% automatic CV results.
% C = 2.^(-0.2:-0.2:-1);
% C = 2^-0.5;
% C = 2.^(0.2:0.2:1);
C = 1; % default 1 works ok

% tolerance of termination criterion
% NOTE: since SVM training is not the most time consuming part (cyclic
% optimization in SCutFBR.1 on the other hand is), we use a small tolerance
% to ensure accurate results
e = 0.0001;


%% Load data

disp('INFO: loading training data')
[y_train, X_train, label_map] = read_sparse_ml(strcat([home_path, '/Data/wise2014-train.libsvm']));
y_train = full(y_train);
y_train = 2*y_train - 1;

disp('INFO: loading testing data')
[~, X_test, ~] = read_sparse_ml(strcat([home_path, '/Data/wise2014-test.libsvm']));

% useful params
[numTrain, numFeat] = size(X_train);
numTest = size(X_test, 1);
numLabel = size(y_train, 2);


%% Train for each cost C and each algorithm of linear SVM

% perform instance-based feature normalization
[X_train, X_test] = feat_norm(X_train, X_test, feat_norm_method);

% add bias to the training and testing data
bias = 1;
X_train = [bias*ones(numTrain, 1), X_train];
X_test = [bias*ones(numTest, 1), X_test];

% initialize prediction matrix and best F1-score vector for each round
pred = zeros(numTest, numLabel, nr_round);
best_F = zeros(1, nr_round);

for count_algo = 1:length(algorithms)
    algo = algorithms{count_algo};
    for count_c = 1:length(C)
        c = C(count_c);
        
        % random seed to enable reproducible results
        rand('seed', random_seed);
        
        % file name for saving results
        save_csv_file_name = strcat([path_to_save_csv, transformation,'_',...
            '[FeatNorm_', feat_norm_method,']_',...
            '[SCutFBR_Tol', num2str(improve_tol),']_',...
            '[SCutFBR_Fold',num2str(nr_fold),']_',...
            '[',algo,']_',...
            '[SVM_C', num2str(c), ']_',...
            '[SVM_e', num2str(e), ']_',...
            '[EnsembleRound', num2str(nr_round),']_',...
            '[InstRatio', num2str(instance_ratio),']_',...
            '[InstReplacement', num2str(instance_replacement),']_',...
            '[FeatRatio', num2str(feature_ratio),']_',...
            '[Seed', num2str(random_seed),']',...
            '.csv']);
        save_model_file_name = strcat([save_csv_file_name(1:end-3),'mat']);
        
        % train random shuffle instace/label data set, which are used
        % to create the final bagging submission
        for round = 1:nr_round
            
            % randomly draw samples
            if instance_replacement == true
                % sampling with replacement
                % This is similar as bagging but which is with instance_ratio = 1.0
                sample_index = randi(numTrain, floor(numTrain*instance_ratio), 1);
            elseif instance_replacement == false
                % sampling without replacement as bagging
                sample_index = randperm(numTrain);
                sample_index = sample_index(1:floor(numTrain*instance_ratio));
            else
                error('instance_replacment can either be logical true or false.');
            end

            % randomly draw features
            feature_index = randperm(numFeat);
            feature_index = feature_index(1:floor(numFeat*feature_ratio));
            % add 1 due to bias term in the first column of feature matrix
            feature_index = [1, feature_index + 1];
            
            % since label order affects the performance of cyclic
            % optimization in SCutFBR.1, we shuffle the label order before
            % optimization to ensure diverse results
            label_index = randperm(numLabel);
            [~, inv_label_index] = sort(label_index);

            % obtain (W, B) for all labels
            [W, B, best_F(round)] = train_fbr_list_mean_F1score(X_train(sample_index, feature_index), y_train(sample_index, label_index),...
                fbr_list, transformation, algo, c, e, nr_fold, improve_tol, verbose);
            
            % make prediction on training data
            y_train_pred_binary = make_prediction(X_train(sample_index, feature_index), transformation, W, B, 'binary');
            F = computeF1score(y_train(sample_index, label_index), y_train_pred_binary, 'mean');
            disp(strcat(['Mean F1-score for the training data using final model : F = ', num2str(F)]));
            
            % make prediction on testing data
            p = make_prediction(X_test(:, feature_index), transformation, W, B, 'bias_raw');
            % inverse label order to get labels aligned
            pred(:,:,round) = p(:,inv_label_index);
            
            % make binary prediction (0/1) using prediction from 1 up to round run
            pred1 = (mean(pred(:,:,1:round),3)>0);
            pred2 = (mean((pred(:,:,1:round)>0),3)>0.5);
            
            % make submission
            % rule1
            save_csv_file_name_rule1 = strcat([save_csv_file_name(1:end-4), '_',...
                '[Fscore', num2str(mean(best_F(1:round)), 5), ']_',...
                '[Round', num2str(round), ']_',...
                'rule1.csv']);
            make_submission(pred1, label_map, save_csv_file_name_rule1);
            % rule2
            save_csv_file_name_rule2 = strcat([save_csv_file_name(1:end-4), '_',...
                '[Fscore', num2str(mean(best_F(1:round)), 5), ']_',...
                '[Round', num2str(round), ']_',...
                'rule2.csv']);
            make_submission(pred2, label_map, save_csv_file_name_rule2);        
        end
        % save model in mat format
        save(save_model_file_name, 'pred', 'label_map', 'best_F', 'c', 'algo');        
    end
end

%% Clean up
matlabpool close
rmpath(genpath(strcat([home_path, '/MATLAB/Opt_Mean_F1score/'])));
rmpath(genpath(strcat([home_path, '/MATLAB/utils/'])));