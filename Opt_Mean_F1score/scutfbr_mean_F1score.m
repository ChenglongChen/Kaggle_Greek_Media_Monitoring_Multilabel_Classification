%
% SCutFBR.1 for optimizing mean F1-score
% 
% Reference:
% 
% [1] David D. Lewis, Yiming Yang, Tony G. Rose, and Fan Li.
% RCV1: A new benchmark collection for text categorization research.
% Journal of Machine Learning Research, 5:361-397, 2004.
% 
% [2] Rong-En Fan and Chih-Jen Lin.
% A Study on Threshold Selection for Multi-label Classification.


function [w, b_list] = scutfbr_mean_F1score(y_train, X_train, fbr_list,...
    transformation, algo, C, e, nr_fold, improve_tol, verbose)

% some useful parameters
[numTrain, ~] = size(X_train);
numLabel = size(y_train, 2);
% numLabel = 10;
% initialize b_list for each fbr in fbr_list and label
b_list = zeros(length(fbr_list), numLabel);

% random shuffle index for cv
perm = randperm(numTrain)';

% cv
% since feature dim increase as chain grows, we use cell for holding
% wegihts. Actually, you can use matrix as well with some handling of the
% indices. But I leave it as it is.
scut_w = cell(1, numLabel);
scut_b = zeros(1, numLabel);
for fold = 1:nr_fold
    time_fold = cputime;
    
    train_id = [1:floor((fold-1)*numTrain/nr_fold) floor(fold*numTrain/nr_fold)+1:numTrain]';
    valid_id = [floor((fold-1)*numTrain/nr_fold)+1:floor(fold*numTrain/nr_fold)]';
    
    % initialize binary classifier for each label
    y_train2 = y_train(perm(train_id),:);
    X_train2 = X_train(perm(train_id),:);
    
    if strcmp(transformation, 'BR')
        parfor j = 1:numLabel
            [scut_w{j}, scut_b(j)] = do_train(y_train2(:,j), X_train2, algo, C, e);
        end
    elseif strcmp(transformation, 'CC')
        % train classifier chains
        parfor j = 1:numLabel
            % augment the features with previous labels 1 to j-1
            % remember to convert labels -1/+1 to 0/1 (necessary?)
            y_augmented = (y_train2(:,1:j-1) > 0);
            [scut_w{j}, scut_b(j)] = do_train(y_train2(:,j), [X_train2, y_augmented], algo, C, e);
        end
    else
        error('Currently only support BR and CC');
    end
    
    % cyclic optimization
    % validation data
    X_valid = X_train(perm(valid_id),:);
    y_valid = y_train(perm(valid_id),:);
    numValid = size(X_valid, 1);
    % make prediction
    wTx = make_prediction(X_valid, transformation, scut_w, scut_b, 'no_bias_raw');
    y_pred = 2*(bsxfun(@plus, wTx, scut_b)>0) - 1;
    y_pred_old = y_pred;
    F_t = [];
    F_t(1) = mean_F1score(y_valid(:,1:numLabel), y_pred);
    disp('Initial F:');
    disp(F_t(1));
    start_F = F_t(1);
    % keep track of iteration number
    t = 1;
    while true
        time_iter = cputime;
        for j = 1:numLabel
            
            time_label = cputime;

            % use the negative sorted version as threshold candidate
            [thresh, thresh_index] = sort(wTx(:,j), 1, 'ascend');
            % special deal with head and tail
            thresh(1) = thresh(1) - eps;
            thresh(end) = thresh(end) + eps;
            % take the negative
            thresh = - thresh;

            % The following part, i.e., cyclic optimization, which tries to
            % find the best mean F1-score and the corresponding threshold
            % seems the most time consuming part (bottleneck).
            % Try to optimize it.
            if strcmp(transformation, 'BR')
                F = cyclic_opt_update_F_BR(y_valid(:,1:numLabel), y_pred, j, thresh_index);
            elseif strcmp(transformation, 'CC')
                F = cyclic_opt_update_F_CC(y_valid(:,1:numLabel), y_pred, j, thresh_index, wTx, scut_w, scut_b);
                % There seems still bug in the C-MEX code which
                % ocassionally causes MATLAB to crash. So, I would
                % recommand you to use the pure MATLAB function above. Or
                % help me debug it :-)
%                 F = cyclic_opt_update_F_CC_mex_Chen(y_valid(:,1:numLabel), y_pred, j, thresh_index, wTx, scut_w, scut_b);
            else
                error('Currently only support BR and CC');
            end
            
            % get the best mean F1-score and the corresponding threshold
            [best_F, cut] = max(F);
%             toc
            
            % modify b
            if best_F > start_F
                if cut == 1 || cut == numValid % i.e., all +1/-1
                    scut_b(j) = thresh(cut);
                else
                    scut_b(j) = (thresh(cut) + thresh(cut+1))/2;
%                     scut_b(j) = thresh(cut+1);
                end
            end
            
            % F for this iteration and label j
            % the same update rule for BR and CC
            y_pred(:,j) = 2*(wTx(:,j) + scut_b(j)>0) - 1;
            
            if (strcmp(transformation, 'CC') && j < numLabel)
                % for CC, the binary prediction corresponding to label j 
                % affects the no_bias_raw and binary prediction of labels
                % j+1, j+2, ... labels.
                % So we have to update prediction for j~numLabel
                
                % BUT, this seems quite time consuming!! [for j = 1, ~9s]
                 
                % update for labels j+1 ~ numLabel
                for jj = j+1:numLabel
                    % no bias raw prediction
                    % augment the features with previous labels 1 to j-1
                    % remember to convert labels -1/+1 to 0/1 (necessary?)
                    delta_y = (y_pred(:,j:jj-1) - y_pred_old(:,j:jj-1))/2;
                    delta_wTx = delta_y * scut_w{jj}(end-jj+j+1:end);
                    wTx(:,jj) = wTx(:,jj) + delta_wTx;
                    % binary prediction
                    y_pred(:,jj) = 2*(wTx(:,jj) + scut_b(jj)>0) - 1;
                end
            end
            % keep the previous prediction
            y_pred_old = y_pred;
            % compute this new mean F1-score
            start_F = mean_F1score(y_valid(:,1:numLabel), y_pred);
            if verbose >= 3
                disp(strcat(['  * CV fold ', num2str(fold),...
                    ' | iteration ', num2str(t),...
                    ' | label ', num2str(j), ' : F = ', num2str(start_F),...
                    ' (', num2str(cputime - time_label, 3), 's)']));
            end
        end
        
        % F for this iteration
        F_t(t+1) = start_F;
        if verbose >= 2
            disp(strcat([' ** CV fold ', num2str(fold),...
                ' | iteration ', num2str(t), ' : F = ', num2str(F_t(t+1)),...
                ' (', num2str(cputime - time_iter, 3), 's)']));
        end
        % check termination criterion
        if (F_t(t+1) - F_t(t))/F_t(1) < improve_tol
%             disp(strcat(['Optimization ends at iteration ', num2str(t)]));
            break;
        else
            % accumulate iteration number
            t = t + 1;
        end
    end
    
    % compare to fbr_list
    if verbose >= 1
        disp(strcat(['*** CV fold ', num2str(fold), ' : F = ', num2str(F_t(end)),...
            ' (', num2str(cputime - time_fold, 3), 's)']));
    end
    for i = 1:length(fbr_list)
        if F_t(end) > fbr_list(i)
            b_list(i,:) = b_list(i,:) + scut_b;
        else
            b_list(i,:) = b_list(i,:) - max(wTx);
        end
    end
end

% final model
b_list = b_list / nr_fold;
w = cell(1, numLabel);
if strcmp(transformation, 'BR')
    parfor j = 1:numLabel
        [w{j}, ~] = do_train(y_train(:,j), X_train, algo, C, e);
    end
elseif strcmp(transformation, 'CC')
    % train classifier chains
    parfor j = 1:numLabel
        % augment the features with previous labels 1 to j-1
        % remember to convert label -1/+1 to 0/1
        y_augmented = (y_train(:,1:j-1) > 0)
        [w{j}, ~] = do_train(y_train(:,j), [X_train, y_augmented], algo, C, e);
    end
else
    error('Currently only support BR and CC');
end

end


function F = cyclic_opt_update_F_BR(y_valid, y_pred, j, thresh_index)
% METHOD 3: using for [approximately 0.1~0.3s]
% It is inspired by scutfbr.m in
% http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/multilabel/rcv1/rcv1_lineart_col.m
% NOTE: The following can only be done in a sequetial manner,
% so do NOT use parfor here


% count tp/fn/fp using labels other than label j to save time
% since they are not affected by the optimization on label j
TP = sum(y_valid(:,[1:j-1,j+1:end]) == +1 & y_pred(:,[1:j-1,j+1:end]) == +1, 2);
FN = sum(y_valid(:,[1:j-1,j+1:end]) == +1 & y_pred(:,[1:j-1,j+1:end]) == -1, 2);
FP = sum(y_valid(:,[1:j-1,j+1:end]) == -1 & y_pred(:,[1:j-1,j+1:end]) == +1, 2);


numValid = size(y_valid(:,j), 1);
F = zeros(1, numValid);

% initialize each instance's tp/fp/fn for label j
tp = double(y_valid(:,j) == +1);
fp = 1 - tp;
fn = zeros(numValid, 1);
% combine with results from other labels
tp = tp + TP;
fp = fp + FP;
fn = fn + FN;
% initialize F1-score for each instance
flag = double(tp ~= 0 | fp ~= 0 | fn ~= 0);
f = (2*tp) ./ (2*tp + fp + fn) .* flag;
f(isnan(f)) = 0;
% initialize mean F1-score
f_mean = mean(f);

% scan through all thresholds
% Note: the following for-loop might be further speeded up using mex code
for i = 1:numValid
    % get the index of this instance
    ii = thresh_index(i);
    % update tp/fp/fn of instance at index ii
    if y_valid(ii,j) == -1
        fp(ii) = fp(ii) - 1;
    else
        tp(ii) = tp(ii) - 1;
        fn(ii) = fn(ii) + 1;
    end
    % update F1-score of instance at index ii
    flag_ii = double(tp(ii) ~= 0 | fp(ii) ~= 0 | fn(ii) ~= 0);
    f_ii_new = (2*tp(ii)) ./ (2*tp(ii) + fp(ii) + fn(ii)) .* flag_ii;
    f_ii_new(isnan(f_ii_new)) = 0;
    % update mean F1-score using the new F1-score of instance
    % at index ii
    f_mean = f_mean + (f_ii_new - f(ii)) / numValid;
    F(i) = f_mean;
end

end


function F = cyclic_opt_update_F_CC(y_valid, y_pred, j, thresh_index, wTx, scut_w, scut_b)
% METHOD 3: using for [approximately 0.1~0.3s]
% It is inspired by scutfbr.m in
% http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/multilabel/rcv1/rcv1_lineart_col.m
% NOTE: The following can only be done in a sequetial manner,
% so do NOT use parfor here

% count tp/fn/fp using labels 1,2,...,j-1 to save time
% since they are not affected by the optimization on label j
TP = sum(y_valid(:,[1:j-1,j+1:end]) == +1 & y_pred(:,[1:j-1,j+1:end]) == +1, 2);
FN = sum(y_valid(:,[1:j-1,j+1:end]) == +1 & y_pred(:,[1:j-1,j+1:end]) == -1, 2);
FP = sum(y_valid(:,[1:j-1,j+1:end]) == -1 & y_pred(:,[1:j-1,j+1:end]) == +1, 2);

TP_j = sum(y_valid(:,j+1:end) == +1 & y_pred(:,j+1:end) == +1, 2);
FN_j = sum(y_valid(:,j+1:end) == +1 & y_pred(:,j+1:end) == -1, 2);
FP_j = sum(y_valid(:,j+1:end) == -1 & y_pred(:,j+1:end) == +1, 2);

numValid = size(y_valid, 1);
numLabel = size(y_valid, 2);
F = zeros(1, numValid);

% initialize each instance's tp/fp/fn for label j
tp = double(y_valid(:,j) == +1);
fp = 1 - tp;
fn = zeros(numValid, 1);
% combine with results from other labels
tp = tp + TP;
fp = fp + FP;
fn = fn + FN;
% initialize F1-score for each instance
flag = double(tp ~= 0 | fp ~= 0 | fn ~= 0);
f = (2*tp) ./ (2*tp + fp + fn) .* flag;
f(isnan(f)) = 0;
% initialize mean F1-score
f_mean = mean(f);

% scan through all thresholds
% Note: the following for-loop might be further speeded up using mex code
y_pred_old = y_pred;
for i = 1:numValid
    % get the index of this instance
    ii = thresh_index(i);
    % update tp/fp/fn of instance at index ii
    if y_valid(ii, j) == -1
        fp(ii) = fp(ii) - 1;
    else
        tp(ii) = tp(ii) - 1;
        fn(ii) = fn(ii) + 1;
    end
    
    % update for labels j+1 ~ numLabel if necessary
    if (y_pred_old(ii,j) == +1 && j < numLabel)
        y_pred(ii,j) = -1;
        for jj = j+1:numLabel
            % no bias raw prediction
            % augment the features with previous labels 1 to j-1
            % remember to convert labels -1/+1 to 0/1 (necessary?)
            delta_y = ( y_pred(ii,j:jj-1) - y_pred_old(ii,j:jj-1) ) / 2;
            delta_wTx = delta_y * scut_w{jj}(end-jj+j+1:end);
            wTx(ii,jj) = wTx(ii,jj) + delta_wTx;
            % binary prediction
            y_pred(ii,jj) = 2*(wTx(ii,jj) + scut_b(jj)>0) - 1;
        end
        % update tp/fn/fp
        delta_tp_ii = sum(y_valid(ii,j+1:end) == +1 & y_pred(ii,j+1:end) == +1) - TP_j(ii);
        delta_fn_ii = sum(y_valid(ii,j+1:end) == +1 & y_pred(ii,j+1:end) == -1) - FN_j(ii);
        delta_fp_ii = sum(y_valid(ii,j+1:end) == -1 & y_pred(ii,j+1:end) == +1) - FP_j(ii);
        tp(ii) = tp(ii) + delta_tp_ii;
        fn(ii) = fn(ii) + delta_fn_ii;
        fp(ii) = fp(ii) + delta_fp_ii;
    end
    
    % update F1-score of instance at index ii
    flag_ii = double(tp(ii) ~= 0 | fp(ii) ~= 0 | fn(ii) ~= 0);
    f_ii_new = (2*tp(ii)) ./ (2*tp(ii) + fp(ii) + fn(ii)) .* flag_ii;
    f_ii_new(isnan(f_ii_new)) = 0;
    % update mean F1-score using the new F1-score of instance
    % at index ii
    f_mean = f_mean + (f_ii_new - f(ii)) / numValid;
    F(i) = f_mean;
end

end


function F = cyclic_opt_update_F_CC_mex_Chen(y_valid, y_pred, j, thresh_index, wTx, scut_w, scut_b)
% for calling the mex function

numLabel = size(y_valid, 2);
w_array = zeros(numLabel, numLabel);
for i = 2:numLabel
    w_array(1:i-1,i) = scut_w{i}(end-i+2:end);
end

F = cyclic_opt_update_F_CC_mex(y_valid, y_pred, wTx, w_array, scut_b, thresh_index, j);

end