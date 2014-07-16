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
w_pos = 1;
w_neg = sum(y == +1)/sum(y == -1);
% If you do not want to use class weights, uncomment the following
w_neg = 1;
switch algo
  case 'l2reg_lr_primal'
    cmd = strcat(['-s 0 -c ', num2str(C), ' -w1 ', num2str(w_pos,2), ' -w-1 ', num2str(w_neg,2), ' -B -1 -e ', num2str(e), ' -q']);
  case 'l2reg_l2loss_dual'
    cmd = strcat(['-s 1 -c ', num2str(C), ' -w1 ', num2str(w_pos,2), ' -w-1 ', num2str(w_neg,2), ' -B -1 -e ', num2str(e), ' -q']);
  case 'l2reg_l2loss_primal'
    cmd = strcat(['-s 2 -c ', num2str(C), ' -w1 ', num2str(w_pos,2), ' -w-1 ', num2str(w_neg,2), ' -B -1 -e ', num2str(e), ' -q']);
  case 'l2reg_l1loss_dual'
    cmd = strcat(['-s 3 -c ', num2str(C), ' -w1 ', num2str(w_pos,2), ' -w-1 ', num2str(w_neg,2), ' -B -1 -e ', num2str(e), ' -q']);
  case 'crammer_singer'
    cmd = strcat(['-s 4 -c ', num2str(C), ' -w1 ', num2str(w_pos,2), ' -w-1 ', num2str(w_neg,2), ' -B -1 -e ', num2str(e), ' -q']);
  case 'l1reg_l2loss_loss'
    cmd = strcat(['-s 5 -c ', num2str(C), ' -w1 ', num2str(w_pos,2), ' -w-1 ', num2str(w_neg,2), ' -B -1 -e ', num2str(e), ' -q']);
  case 'l1reg_lr'
    cmd = strcat(['-s 6 -c ', num2str(C), ' -w1 ', num2str(w_pos,2), ' -w-1 ', num2str(w_neg,2), ' -B -1 -e ', num2str(e), ' -q']);
  case 'l2reg_lr_dual'
    cmd = strcat(['-s 7 -c ', num2str(C), ' -w1 ', num2str(w_pos,2), ' -w-1 ', num2str(w_neg,2), ' -B -1 -e ', num2str(e), ' -q']);
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