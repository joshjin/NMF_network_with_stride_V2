function result=expand9x9(data_mat)
[len,~] = size(data_mat);
disp(len);
result = zeros(len * 20 * 20,9,9);
data_mat = reshape(data_mat,len,28,28);
for idx = 0:len-1
    for i = 0:28-9
        for j = 0:28-9
            result(1 + idx * (28 - 8) * (28 - 8) + i * (28 - 8) + j,:,:) = data_mat(1+idx,i+1:i+9,j+1:j+9);
        end
    end
end
end