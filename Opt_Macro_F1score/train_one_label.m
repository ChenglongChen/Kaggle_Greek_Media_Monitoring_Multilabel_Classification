%
% train_one_label
%

function [w, b] = train_one_label(X_train, proby, algo, nr_fold)

fbr_list = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8];
% fbr_list = 0.05:0.05:0.95;
l = size(proby, 1);

perm = randperm(l)';

f_list = zeros(size(fbr_list));

% cv
for fold = 1:nr_fold
  train_id = [1:floor((fold-1)*l/nr_fold) floor(fold*l/nr_fold)+1:l]';
  valid_id = [floor((fold-1)*l/nr_fold)+1:floor(fold*l/nr_fold)]';

  y = proby(perm(train_id));
  validy = proby(perm(valid_id));

  % scutfbr
  [scutfbr_w, scutfbr_b_list] = scutfbr(y, X_train(perm(train_id),:), fbr_list, algo, nr_fold);
  wTx = X_train(perm(valid_id),:)*scutfbr_w; % +b

  for i = 1:size(fbr_list, 2)
    F = fmeasure(validy, 2*(wTx > -scutfbr_b_list(i))-1);
    f_list(i) = f_list(i) + F;
  end
end

best_fbr = fbr_list(find(f_list == max(f_list), 1, 'last'));
if max(f_list) == 0
  best_fbr = min(fbr_list);
  fprintf(1, 'INFO: train_one_label: F all 0\n');
end

% final model
[w, b] = scutfbr(proby, X_train, best_fbr, algo, nr_fold);
fprintf(1, 'INFO: train_one_label: best_fbr %.1f\n', best_fbr);

end