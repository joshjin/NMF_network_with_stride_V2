function [sig] = ica_step_in_test(data, A ,patch_size)

[data_len,feat_len] = size(data);
disp('data in ica step');
disp(size(data));
data_ = reshape(data,patch_size,patch_size,data_len/(patch_size*patch_size),feat_len);
disp('data_ in ica step');
disp(size(data_));
V = merge_local_feat_5x5_new(data_)';
disp('A in ica step');
disp(size(A));
disp('V in ica step');
disp(size(V));
sig = pinv(A) * V;
disp('sig size');
disp(size(sig))
end