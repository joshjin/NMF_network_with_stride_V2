function result=expand6x6(data_mat)
[len,~] = size(data_mat);
disp(len);
result = zeros(len * 23 * 23,6,6);
data_mat = reshape(data_mat,len,28,28);
for idx = 0:len-1
    for i = 0:28-6
        for j = 0:28-6
            result(1 + idx * (28 - 5) * (28 - 5) + i * (28 - 5) + j,:,:) = data_mat(1+idx,i+1:i+6,j+1:j+6);
        end
    end
end
end