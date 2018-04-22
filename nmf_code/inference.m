function [output,activations] = inference(model,input)
% Do forward propagation through the network to get the activation
% at each layer, and the final output

num_layers = numel(model.layers);
activations = cell(num_layers,1);

% TODO: FORWARD PROPAGATION CODE
in = input;
for i=1:num_layers
    layer = model.layers(i);
    in = layer.fwd_fn(in, layer.params, layer.hyper_params, false);
    activations{i} = in;
end

output = activations{end};
