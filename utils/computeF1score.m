%
% Calculate singleLabel/macro/micro/mean F1-score for two-class (-1/+1) problem
% 

function F = computeF1score(original, predict, type)

if strcmpi(type, 'singleLabel')
    tp = full(sum(original(:) == +1 & predict(:) == +1));
    fn = full(sum(original(:) == +1 & predict(:) == -1));
    fp = full(sum(original(:) == -1 & predict(:) == +1));
elseif strcmpi(type, 'macro')
    tp = full(sum(original == +1 & predict == +1, 1));
    fn = full(sum(original == +1 & predict == -1, 1));
    fp = full(sum(original == -1 & predict == +1, 1));
elseif strcmpi(type, 'micro')
    tp = full(sum(sum(original == +1 & predict == +1, 1), 2));
    fn = full(sum(sum(original == +1 & predict == -1, 1), 2));
    fp = full(sum(sum(original == -1 & predict == +1, 1), 2));
elseif strcmpi(type, 'mean')
    tp = full(sum(original == +1 & predict == +1, 2));
    fn = full(sum(original == +1 & predict == -1, 2));
    fp = full(sum(original == -1 & predict == +1, 2));
else
    msg = strcat([ 'Wrong type of F1-score.\n',...
                   'type arg can either be singleLabel, macro, micro, or mean\n']);
    error('myApp:argChk', msg);
end

flag = double(tp ~= 0 | fp ~= 0 | fn ~= 0);
F = (2*tp) ./ (2*tp + fp + fn) .* flag;
F(isnan(F)) = 0;
F = mean(F);

end