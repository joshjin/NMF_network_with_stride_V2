% conv-liked cut pieces
% input: data_mat: n x 784
function result=expand3x3(data_mat)
[len,~] = size(data_mat);
disp(len);
result = zeros(len * 26 * 26,3,3);
data_mat = reshape(data_mat,len,28,28);
for idx = 0:len-1
    for i = 0:28-3
        for j = 0:28-3
            result(1 + idx * (28 - 2) * (28 - 2) + i * (28 - 2) + j,:,:) = data_mat(1+idx,i+1:i+3,j+1:j+3);
        end
    end
end
end