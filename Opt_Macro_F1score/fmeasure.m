%
% Calculate F_{1.0) measure for two-class problem
% classes are +1, -1
%

function F = fmeasure(original, predict)

F = 0;

tp = full(sum(original == +1 & predict == +1));
fn = full(sum(original == +1 & predict == -1));
fp = full(sum(original == -1 & predict == +1));

if tp ~= 0 || fp ~= 0 || fn ~= 0
  F = (2*tp) / (2*tp + fp + fn);
end
end