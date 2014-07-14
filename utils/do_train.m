%
% train linear SVM via liblinear
%

function [w, b] = do_train(y, x, algo, C, e)

if nargin == 3
    C = 1;
    e = 0.5;
elseif nargin == 4
    e = 0.5;
end

% class weights
w1 = 1;
w2 = sum(y == +1)/sum(y == -1);
% If you do not want to use class weights, uncomment the following
w2 = 1;
switch algo
  case 'lr'
    cmd = strcat(['-s 0 -c ', num2str(C), ' -w1 ', num2str(w1,2), ' -w2 ', num2str(w2,2), ' -B -1 -e ', num2str(e), ' -q']);
  case 'l2svm_dual'
    cmd = strcat(['-s 1 -c ', num2str(C), ' -w1 ', num2str(w1,2), ' -w2 ', num2str(w2,2), ' -B -1 -e ', num2str(e), ' -q']);
  case 'l2svm'
    cmd = strcat(['-s 2 -c ', num2str(C), ' -w1 ', num2str(w1,2), ' -w2 ', num2str(w2,2), ' -B -1 -e ', num2str(e), ' -q']);
  case 'l1svm_dual'
    cmd = strcat(['-s 3 -c ', num2str(C), ' -w1 ', num2str(w1,2), ' -w2 ', num2str(w2,2), ' -B -1 -e ', num2str(e), ' -q']);
  otherwise
    disp('INFO: Unkown option')
end

model = train(y, x, cmd);

w = model.w(:);
b = model.bias;
% b = 0;

if model.Label(1) == -1
  w = -w;
  b = -b;
end

end