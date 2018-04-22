function result=expand10x10(data_mat)
[len,~] = size(data_mat);
disp(len);
result = zeros(len * 19 * 19,10,10);
data_mat = reshape(data_mat,len,28,28);
for idx = 0:len-1
    for i = 0:28-11
        for j = 0:28-11
            result(1 + idx * (28 - 10) * (28 - 10) + i * (28 - 10) + j,:,:) = data_mat(1+idx,i+1:i+10,j+1:j+10);
        end
    end
end
end