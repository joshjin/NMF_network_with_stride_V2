clear

size_before_flatten = 6;
params_linear_1 = struct('W', rand(144,2880), 'b', rand(144,1));
params_linear_2 = struct('W', rand(10,144), 'b', rand(10,1));

% read data into var: data
fid = fopen( 't10k-images.idx3-ubyte', 'r' );
data = fread( fid, 'uint8' );
fclose(fid);
data = data(17:end);
data = reshape( data, 28*28, 10000 )';
short_data = data(1:256,:);
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

disp('Expanding Images to 9x9 Patch');
% V = expand3x3(data);
V = expand9x9(short_data);
[len,~,~] = size(V);
disp(size(V));

V = reshape(V,len,81);
V = V + (V < 0.95) * 1e-3;
disp('NNMF Optimizing Step 1');
[W1,H1] = nmf_step_9x9(V,24,20);


