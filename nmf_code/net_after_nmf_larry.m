% neural net script after training through 10-layer nmf
addpath pcode;
addpath layers;
addpath tools;

sample_size = 1000;

file = load('icasig3_60_100_120_STRIDE4.mat');
H10 = file.ica_param.icasig3;
disp("H10 size: ");
disp(size(H10));
[s1, s2] = size(H10);
s0 = 11;
feat_num = 100;
% flaten
% fully connected
% loss function
input = reshape(H10, feat_num, s0, s0, s2/s0/s0); 
input = reshape(input, s0, s0, feat_num, s2/s0/s0);    % new size = 6x6x80x256
l = [
    init_layer('flatten',struct('num_dims',4))        
    init_layer('linear',struct('num_in',s0*s0*feat_num,'num_out',2880))
    init_layer('tanh', []);
%     init_layer('linear',struct('num_in',2880,'num_out',1440))
%     init_layer('tanh', []);
%     init_layer('linear',struct('num_in',1440,'num_out',720))
%     init_layer('tanh', []);
    init_layer('linear',struct('num_in',2880,'num_out',10))
    init_layer('tanh', []);
	init_layer('softmax',[])
    ];

% Learning rate
lr = 0.2;
% Weight decay
wd = .001;
% Batch size
batch_size = 10;

model = init_model(l,[s0 s0 feat_num],10,true);



% Saved model name
save_file = 'model_2nd_net.mat';

params = struct('learning_rate',lr,'weight_decay',wd,'batch_size',batch_size,'save_file',save_file,'epoch',0);

numIters = size(input,4) / batch_size;

max_epochs = 8;

% read label into var: label
fid = fopen( 't10k-labels.idx1-ubyte', 'r' );
label = fread( fid, 'uint8' );
fclose(fid);
label = label(9:end);
label = label(1:sample_size);
label(label==0)=10;

for epoch=1:max_epochs
    params.epoch = epoch;
    [model, loss] = train(model,input,label,params,numIters);
end
    
