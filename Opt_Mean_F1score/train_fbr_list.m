%
% train_fbr_list
% 
% Reference:
% 
% [1] David D. Lewis, Yiming Yang, Tony G. Rose, and Fan Li.
% RCV1: A new benchmark collection for text categorization research.
% Journal of Machine Learning Research, 5:361-397, 2004.
% 
% [2] Rong-En Fan and Chih-Jen Lin.
% A Study on Threshold Selection for Multi-label Classification.


function [w, b, best_F] = train_fbr_list(X_train, y_train, fbr_list,...
    transformation, algo, C, e, nr_fold, improve_tol, verbose)

% some useful parameters
[numTrain, ~] = size(X_train);
numLabel = size(y_train, 2);

% random shuffle index for cv
perm = randperm(numTrain)';

f_list = zeros(nr_fold, length(fbr_list));

% Note: I once tried to use parfor here, so I segmented out each for-loop
% block. The speedup is lower than using parfor in scutfbr_mean_F1score.
% So I abandoned it soon. Since then the code here leaves unchanged,
% though it can be cleaned up (i.e., merge the 3 for-loop blocks).

% cv index 
train_id = cell(1,5);
valid_id = cell(1,5);
for fold = 1:nr_fold
    train_id{fold} = [1:floor((fold-1)*numTrain/nr_fold) floor(fold*numTrain/nr_fold)+1:numTrain]';
    valid_id{fold} = [floor((fold-1)*numTrain/nr_fold)+1:floor(fold*numTrain/nr_fold)]';
end

% cv for each fold
scutfbr_w = cell(1, nr_fold);
scutfbr_b_list = zeros(length(fbr_list), numLabel, nr_fold);
for fold = 1:nr_fold  % I tryied parfor here
    y_train2 = y_train(perm(train_id{fold}),:);
    X_train2 = X_train(perm(train_id{fold}),:);
    % scutfbr_mean_F1score
    [scutfbr_w{fold}, scutfbr_b_list(:,:,fold)] = scutfbr_mean_F1score(y_train2, X_train2,...
        fbr_list, transformation, algo, C, e, nr_fold, improve_tol, verbose);
end

% compute mean f score for each fold and each fbr
for fold = 1:nr_fold
    X_valid = X_train(perm(valid_id{fold}),:);
    y_valid = y_train(perm(valid_id{fold}),:);
    for i = 1:length(fbr_list)
        % make prediction
        y_pred = make_prediction(X_valid, transformation,...
            scutfbr_w{fold}, squeeze(scutfbr_b_list(i,:,fold)), 'binary');
        f_list(fold, i) = mean_F1score(y_valid, y_pred);
    end
end
f_list = mean(f_list);

% find the best fbr
best_F = max(f_list);
best_fbr = fbr_list(find(f_list == best_F, 1, 'last'));
if best_F == 0
    best_fbr = min(fbr_list);
    fprintf(1, 'INFO: train_fbr_list: F all 0\n');
end

% final model
[w, b] = scutfbr_mean_F1score(y_train, X_train, best_fbr,...
    transformation, algo, C, e, nr_fold, improve_tol, verbose);

fprintf(1, 'INFO: train_fbr_list: best_fbr %.1f\n', best_fbr);

end