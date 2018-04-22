addpath pcode;
addpath layers;

my_model = load('model_2nd_net.mat');
model = my_model.model;
file = load('icasig3_60_100_120_STRIDE4.mat');
ica_param = file.ica_param;

batch_size = 100;

i = 1;   % set of testing

% read data into var: data
fid = fopen( 't10k-images.idx3-ubyte', 'r' );
data_temp = fread( fid, 'uint8' );
fclose(fid);
data_temp = data_temp(17:end);
data_temp = reshape( data_temp, 28*28, 10000 )';
data_test = data_temp(1000*(i-1)+1:1000*i,:);

% read label into var: label
fid = fopen( 't10k-labels.idx1-ubyte', 'r' );
label_temp = fread( fid, 'uint8' );
fclose(fid);
label_temp = label_temp(9:end);
label_test = label_temp(1000*(i-1)+1:1000*i);
label_test(label_test==0)=10;

nV = expand6x6(data_test);
[len,~,~] = size(nV);
nV = reshape(nV,len,36);

% nH2= nmf_step_in_test(nH1', nmf_param.W2,24);
sig1 = ica_step_in_test(nV, ica_param.A1,23);
disp('size of sig1');
disp(size(sig1));
sig2 = ica_step_in_test(sig1', ica_param.A2,19);
disp('size of sig2');
disp(size(sig2));
% sig3 = ica_step_in_test(sig2', ica_param.A3,15);
% disp('size of sig3');
% disp(size(sig3));

[s1, s2] = size(sig2);
s0 = 15;
feat_num = 178;
% flaten
% fully connected
% loss function
input = reshape(sig2, feat_num, s0, s0, s2/s0/s0); 
input = reshape(input, s0, s0, feat_num, s2/s0/s0);    % new size = 6x6x80x256

overall_right = 0;
iters = size(input,4) / batch_size;
for i=1:iters
    batch = input(:,:,:,(i-1)*batch_size+1:i*batch_size);
    batch_label = label_test((i-1)*batch_size+1:i*batch_size,:);
    [output,~] = inference_(model,batch);
    [~,I] = max(output);
    comp = I' == batch_label;
    overall_right = overall_right + sum(comp);
    accuracy = 100 * overall_right / (i*batch_size);
    disp(accuracy);
end