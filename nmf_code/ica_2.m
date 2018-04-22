%% PCA demo -- digit
clear

addpath('package/FastICA_25')

% size_before_flatten = 6;
% params_linear_1 = struct('W', rand(144,2880), 'b', rand(144,1));
% params_linear_2 = struct('W', rand(10,144), 'b', rand(10,1));

% read data into var: data
fid = fopen( 't10k-images.idx3-ubyte', 'r' );
data = fread( fid, 'uint8' );
fclose(fid);
data = data(17:end);
data = reshape( data, 28*28, 10000 )';
short_data = data(1:1000,:);
disp('data size');
disp(size(data));
disp('short data size');
disp(size(short_data));

% read label into var: label
fid = fopen( 't10k-labels.idx1-ubyte', 'r' );
label = fread( fid, 'uint8' );
fclose(fid);
label = label(9:end);
disp('label size');
disp(size(label));
label_mat = turn_label_to_mat(label);

show_data = reshape(data,10000,28,28);
disp('show data size');
disp(size(show_data));

disp('Expanding Images to 5x5 Patch');
V = data;
% V = expand3x3(data);
% V = expand6x6(short_data);
disp(size(V));
[len,~,~] = size(V);
% V = reshape(V,len,36);

disp('ICA Optimizing Step 1')
[icasig1, A1, W1, tmp1] = ica_step_new(V,28,80,0,2,5,0);

% disp('ICA Optimizing Step 1')
% [icasig1, A1, W1, tmp1] = ica_step_new(V,23,80,0,2,5,1);
% disp('ICA Optimizing Step 1')
% [icasig2, A2, W2, tmp2] = ica_step_new(icasig1',19,80,0,2,5,0);

% disp(size(W1));
% disp(size(icasig1));
% tmp11 = A1 * icasig1;
% tmp12 = W1' * icasig1;
% disp(size(tmp1));
% disp(size(tmp11));i
% disp(size(tmp12));

% disp('ICA Optimizing Step 2')
% [icasig2, A2, W2, tmp2] = ica_step9x9(icasig1',15,60,0);
% tmp21 = A2 * icasig2;
% figure, subplot(2,1,1); imshow(tmp2(1:80,1:80)); subplot(2,1,2); imshow(tmp21(1:80,1:80));

% disp('ICA Optimizing Step 3')
% [icasig3, A3, W3, tmp3] = ica_step9x9(icasig2',15,100,1);
% tmp31 = A3 * icasig3;
% figure, subplot(2,1,1); imshow(tmp3(1:80,1:80)); subplot(2,1,2); imshow(tmp31(1:80,1:80));

ica_param = struct('A1',A1,'tmp1',tmp1,'icasig1',icasig1);
save('icasig3_60_100_120_STRIDE4.mat','ica_param');

% tmp21 = A2 * icasig2;
% tmp22 = W2' * icasig2;

% disp('ICA Optimizing Step 3')
% [icasig3, A3, W3, tmp3] = ica_step9x9(icasig2',20,60,0);

% disp('NNMF Optimizing Step 1')
% [W1,H1] = nmf_step(V,26,20);
% disp('NNMF Optimizing Step 2')
% [W2,H2] = nmf_step(H1',24,20);
% disp('NNMF Optimizing Step 3')
% [W3,H3] = nmf_step(H2',22,20);
% disp('NNMF Optimizing Step 4')
% [W4,H4] = nmf_step(H3',20,40);
% disp('NNMF Optimizing Step 5')
% [W5,H5] = nmf_step(H4',18,40);
% disp('NNMF Optimizing Step 6')
% [W6,H6] = nmf_step(H5',16,40);
% disp('NNMF Optimizing Step 7')
% [W7,H7] = nmf_step(H6',14,80);
% disp('NNMF Optimizing Step 8')
% [W8,H8] = nmf_step(H7',12,80);
% disp('NNMF Optimizing Step 9')
% [W9,H9] = nmf_step(H8',10,80);
% disp('NNMF Optimizing Step 10')
% [W10,H10] = nmf_step(H9',8,80);
% 
