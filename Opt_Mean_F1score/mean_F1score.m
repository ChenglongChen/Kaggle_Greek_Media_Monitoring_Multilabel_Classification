%
% Calculate mean F1-score for two-class problem
% classes are +1, -1
%

function F = mean_F1score(original, predict)

tp = full(sum(original == +1 & predict == +1, 2));
fn = full(sum(original == +1 & predict == -1, 2));
fp = full(sum(original == -1 & predict == +1, 2));

F = computeF1score(tp, fn, fp);

end


%
% Calculate mean F1-score for two-class problem using tp, fn, and fp
%

function F = computeF1score(tp, fn, fp)

% METHOD 1: vectorization
flag = double(tp ~= 0 | fp ~= 0 | fn ~= 0);
F = (2*tp) ./ (2*tp + fp + fn) .* flag;
F(isnan(F)) = 0;
F = mean(F);

% METHOD 2: for-loop
% numTrain = size(tp, 1);
% F = zeros(1, numTrain);
% for i = 1:numTrain
%     if tp(i) ~= 0 || fp(i) ~= 0 || fn(i) ~= 0
%         F(i) = (2*tp(i)) / (2*tp(i) + fp(i) + fn(i));
%     end
% end
% F = mean(F);

end