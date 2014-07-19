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

% choose the right type of SVM
algorithms = { 'l2reg_lr_primal',...
               'l2reg_l2loss_dual',...
               'l2reg_l2loss_primal',...
               'l2reg_l1loss_dual',...
               'crammer_singer_svc',...
               'l1reg_l2loss_svc',...
               'l1reg_lr',...
               'l2reg_lr_dual',...
               };

s = find(strcmp(algo, algorithms)) - 1;
if isempty(s)
    error('INFO: Unkown option')
end

% class weights
w_pos = 1;
w_neg = sum(y == +1)/sum(y == -1);
% If you do not want to use class weights, uncomment the following
w_neg = 1;
cmd = strcat(['-s ', num2str(s), ' -c ', num2str(C), ' -w1 ', num2str(w_pos,2), ' -w-1 ', num2str(w_neg,2), ' -B -1 -e ', num2str(e), ' -q']);
model = train(y, x, cmd);

w = model.w(:);
% b = model.bias;
b = 0;

if model.Label(1) == -1
  w = -w;
  b = -b;
end

end