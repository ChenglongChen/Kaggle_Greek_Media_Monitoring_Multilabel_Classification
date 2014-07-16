% 
% save prediction in csv format
% 

function make_submission(pre, rcv1_map, save_file_name)

f = fopen(save_file_name, 'w');
fprintf(f,'%s,%s\n','ArticleId','Labels');

start_id = 64857;
ArticleId = (1:size(pre,1)) + start_id;
for i = 1:size(pre,1)
    fprintf(f,'%s,', num2str(ArticleId(i)));
    this_label = rcv1_map(pre(i,:)==1);
    if isempty(this_label)
        this_label = 103;
    end
    for j = 1:length(this_label)
        fprintf(f,' %s', num2str(this_label(j)));
    end
    fprintf(f,'\n');
end
fclose(f);

end