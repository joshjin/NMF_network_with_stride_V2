function [output, dv_input, grad] = fn_sigmoid(input, params, hyper_params, backprop, dv_output)
% sigmoid activation

output = 1./(1+exp(-input));

dv_input = [];
grad = struct('W',[],'b',[]);

if backprop
    dv_input = (exp(-dv_output))./((1+exp(-dv_output)).^2);
end