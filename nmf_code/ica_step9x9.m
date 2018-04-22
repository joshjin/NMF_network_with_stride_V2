function [icasig, A, W, V]=ica_step9x9(data,patch_size,feat_out,iffeat)
disp('size of the input data for nmf_step');
disp(size(data));
[data_len,feat_len] = size(data);
data_ = reshape(data,patch_size,patch_size,data_len/(patch_size*patch_size),feat_len);
disp('reshaped data');
disp(size(data_));
% V = data';
% disp('V size');
% disp(size(V));
V = merge_local_feat_new(data_,10,5)';
disp('size of V after merge feature');
disp(size(V));
if iffeat
    [icasig, A, W] = fastica(V, 'numOfIC', feat_out);
else
    [icasig, A, W] = fastica(V);
end
% [W,H] = (V',feat_out);
end