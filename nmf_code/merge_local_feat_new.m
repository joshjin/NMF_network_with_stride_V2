function [result] = merge_local_feat_new(data,stride,filter_size)
filter_size_square = filter_size*filter_size;
[patch_size,~,data_len,feat_len] = size(data);
patched_size = patch_size - filter_size + 1;
disp(size(data));
result = zeros(data_len * patched_size * patched_size, floor(feat_len / stride) * filter_size_square);
disp('size of result');
disp(size(result));
for idx = 0:data_len-1
    for i = 0:(patch_size - filter_size)
        for j = 0:(patch_size - filter_size)
%             disp(size(data(i+1:i+3,j+1:j+3,idx+1,:)));
%             disp(size(result(1 + idx * (patch_size - 2) * (patch_size - 2) + i * (patch_size - 2) + j,:)));
            result(1 + idx * patched_size * patched_size + i * patched_size + j,:) = ...
                reshape(data(i+1:i+filter_size,j+1:j+filter_size,idx+1,1:stride:feat_len),1,floor(feat_len / stride) * filter_size_square);
        end
    end
end
end

