function [W,H] = nmf_step_5x5(data,patch_size,feat_out)
disp('size of the input data for nmf_step');
disp(size(data));
[data_len,feat_len] = size(data);
% data = reshape(data,patch_size,patch_size,data_len/(patch_size*patch_size),feat_len);
% disp('reshaped data');
% disp(size(data));
[W,H] = nnmf(data',feat_out);
end

