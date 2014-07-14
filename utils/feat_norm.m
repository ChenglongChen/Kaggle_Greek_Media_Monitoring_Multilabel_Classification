%
% perform instance-based feature normalization
% can be simply extended

function [X_train, X_test] = feat_norm(X_train, X_test, method)

if strcmp(method, 'binary')
    
    % training data
    X_train = double(X_train>0);
    % testing data
    X_test = double(X_test>0);
    
elseif strcmp(method, 'binary_feat_norm')
    
    % training data
    [numTrain, numFeat] = size(X_train);
    X_train = double(X_train>0);
    new = 1./sqrt(sum(X_train.^2, 1));
    % use the sparsity property
    [i, j] = find(X_train);
    s = new(j);
    X_train = sparse(i, j, s, numTrain, numFeat);

    % testing data
    [numTest, numFeat] = size(X_test);
    X_test = double(X_test>0);
    new = 1./sqrt(sum(X_test.^2, 1));
    % use the sparsity property
    [i, j] = find(X_test);
    s = new(j);
    X_test = sparse(i, j, s, numTest, numFeat);
    
elseif strcmp(method, 'binary_inst_norm')
    
    % training data
    [numTrain, numFeat] = size(X_train);
    X_train = double(X_train>0);
    new = 1./sqrt(sum(X_train.^2, 2));
    % use the sparsity property
    [i, j] = find(X_train);
    s = new(i);
    X_train = sparse(i, j, s, numTrain, numFeat);

    % testing data
    [numTest, numFeat] = size(X_test);
    X_test = double(X_test>0);
    new = 1./sqrt(sum(X_test.^2, 2));
    % use the sparsity property
    [i, j] = find(X_test);
    s = new(i);
    X_test = sparse(i, j, s, numTest, numFeat);
    
elseif strcmp(method, 'log')
    
elseif ~strcmp(method, 'raw')
    error('Must be either: binary, log, or raw');
end

end