% ----------------------------------------------------------------------
% input: in_height x in_width x num_channels x batch_size
% output: out_height x out_width x num_filters x batch_size
% hyper parameters: (stride, padding for further work)
% params.W: filter_height x filter_width x filter_depth x num_filters
% params.b: num_filters x 1
% dv_output: same as output
% dv_input: same as input
% grad.W: same as params.W
% grad.b: same as params.b
% ----------------------------------------------------------------------

function [output, dv_input, grad] = fn_conv(input, params, hyper_params, backprop, dv_output)


[~,~,num_channels,batch_size] = size(input);
[~,~,filter_depth,num_filters] = size(params.W);
assert(filter_depth == num_channels, 'Filter depth does not match number of input channels');

out_height = size(input,1) - size(params.W,1) + 1;
out_width = size(input,2) - size(params.W,2) + 1;
output = zeros(out_height,out_width,num_filters,batch_size);

% TODO: FORWARD CODE
for bs=1:batch_size
    for nf=1:num_filters
        conv_input = zeros(out_height, out_width);
        for nc=1:num_channels
            conv_input  = conv_input + conv2(input(:, :, nc,bs), params.W(:,:,nc,nf), 'valid');
        end
        output(:, :, nf, bs) = conv_input + params.b(nf);
    end
end


dv_input = [];
grad = struct('W',[],'b',[]);
if backprop
	dv_input = zeros(size(input));
	grad.W = zeros(size(params.W));
	grad.b = zeros(size(params.b));
	% TODO: BACKPROP CODE
    for bs=1:batch_size
        for nf=1:num_filters
            for nc=1:num_channels
                dv_input(:,:,nc,bs) = dv_input(:,:,nc,bs) + conv2(dv_output(:,:,nf,bs), rot90(params.W(:,:,nc,nf), 2), 'full');
                grad.W(:,:,nc,nf) = grad.W(:,:,nc,nf) + rot90(conv2(input(:, :, nc,bs), rot90(dv_output(:,:,nf,bs), 2), 'valid'), 2);
            end
        end
    end
    
    for nf=1:num_filters
        grad.b(nf) = sum(sum(sum(dv_output(:,:,nf,:))));
    end
end




