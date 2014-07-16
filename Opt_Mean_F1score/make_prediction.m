%
% make prediction
%

function y = make_prediction(X, transformation, w, b, type)

[numSample, numLabel] = size(X);
% raw prediction without bias term
wTx = zeros(numSample, numLabel);
if strcmp(transformation, 'BR')
    for j = 1:numLabel
        wTx(:,j) = X * w{j};
    end
elseif strcmp(transformation, 'CC')
    for j = 1:numLabel
        % augment the features with previous labels 1 to j-1
        y_augmented = bsxfun(@plus, wTx(:,1:j-1), b(1:j-1));
        % remember to convert to 0/1
        y_augmented = (y_augmented > 0);
        wTx(:,j) = [X, y_augmented] * w{j};
    end
else
    error('Currently only support BR and CC.');
end

if ~strcmp(type, 'no_bias_raw')
    % add bias term
    y = bsxfun(@plus, wTx, b(:)');
else
    % no bias term
    y = wTx;
end

if strcmp(type, 'binary')
    % convert to binary -1/+1
    y = 2*(y>0) - 1;
elseif ~strcmp(type, 'no_bias_raw') && ~strcmp(type, 'bias_raw')
    msg = strcat([ 'argument type can either be no_bias_raw, bias_raw or binary:\n',...
                   'no_bias_raw: y = w^T * x\n',...
                   'bias_raw: y = w^T * x + b\n',...
                   'binary: y = 2*(w^T *x + b > 0) - 1 \\in {-1,+1}']);
    error('myApp:argChk', msg);
end

end