function [result] = merge_local_feat_5x5_new(data)
[patch_size,~,data_len,feat_len] = size(data);
disp(size(data));
result = zeros(data_len * (patch_size - 4) * (patch_size - 4), floor(feat_len / 4) * 25);
disp('size of result');
disp(size(result));
for idx = 0:data_len-1
    for i = 0:(patch_size - 5)
        for j = 0:(patch_size - 5)
%             disp(size(data(i+1:i+3,j+1:j+3,idx+1,:)));
%             disp(size(result(1 + idx * (patch_size - 2) * (patch_size - 2) + i * (patch_size - 2) + j,:)));
            result(1 + idx * (patch_size - 4) * (patch_size - 4) + i * (patch_size - 4) + j,:) = ...
                reshape(data(i+1:i+5,j+1:j+5,idx+1,1:4:feat_len),1,floor(feat_len / 4) * 25);
        end
    end
end
end

