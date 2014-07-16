%
% scutfbr.1
%

function [w, b_list] = scutfbr(proby, probx, fbr_list, algo, nr_fold)
b_list = zeros(size(fbr_list));

l = size(proby, 1);

perm = randperm(l)';

% cv
for fold = 1:nr_fold
  train_id = [1:floor((fold-1)*l/nr_fold) floor(fold*l/nr_fold)+1:l]';
  valid_id = [floor((fold-1)*l/nr_fold)+1:floor(fold*l/nr_fold)]';

  y = proby(perm(train_id));
  validy = proby(perm(valid_id));

  % scut
 [scut_w, scut_b] = do_train(y, probx(perm(train_id),:), algo);

  wTx = probx(perm(valid_id),:)*scut_w;

  start_F = fmeasure(validy, 2*(wTx > -scut_b)-1);

  [sorted_wTx, sorted_wTx_index] = sort(wTx, 1, 'ascend'); 

  tp = sum(validy == 1); fp = size(validy,1)-tp; fn = 0;
  cut = -1;
  best_F = (2*tp) / (2*tp + fp + fn);
  for i = 1:size(validy, 1)
    if validy(sorted_wTx_index(i)) == -1,
      fp = fp -1;
    else
      tp = tp -1; fn = fn + 1;
    end
    F = (2*tp) / (2*tp + fp + fn);
    
    if F >= best_F
      best_F = F;
      cut = i;
    end
  end

  % modify b
  if best_F > start_F
    if cut == -1 % i.e., all +1
      scut_b = - (sorted_wTx(1)-eps); % predict all +1
    elseif cut == size(validy, 1)
      scut_b = - (sorted_wTx(size(validy, 1))+eps);
    else
      scut_b = - (sorted_wTx(cut) + sorted_wTx(cut+1))/2;
    end
  end

  F = fmeasure(validy, 2*(wTx > -scut_b)-1);

  % compare to fbr_list
  for i = 1:size(fbr_list, 2)
    if F > fbr_list(i)
      b_list(i) = b_list(i) + scut_b;
    else
      b_list(i) = b_list(i) - max(wTx);
    end
  end
end

% final model
b_list = b_list / nr_fold;
[w, junk] = do_train(proby, probx, algo);

end
