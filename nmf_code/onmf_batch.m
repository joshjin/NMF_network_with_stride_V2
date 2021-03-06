% TODO:  check   1) implement onmf with general diverence as introduced in the paper
%                            2) vectorize equation

% Online Nonnegative Matrix Factorization with General Divergences

% <Inputs>
%        V : Input data matrix (m x n)
%        k : Target low-rank
%
%        (Below are parameters pre-defined)
%        
%        MAX_ITER : Maximum number of iterations. Default is 100.
%        MIN_ITER : Minimum number of iterations. Default is 20.
%        MAX_TIME : Maximum amount of time in seconds. Default is 100,000.
%        W_INIT : (m x k) initial value for W.
%        H_INIT : (k x n) initial value for H.
%        TOL : Stopping tolerance. Default is 1e-3. If you want to obtain a more accurate solution, decrease TOL and increase MAX_ITER at the same time.
% <Outputs>
%        W : Obtained basis matrix (m x k)
%        H : Obtained coefficients matrix (k x n)

% Here we set V to be known instead of a stream of input
function [W,H] = onmf_batch(V, k)

[m, n] = size(V);
batch_size = 5;

% Default configuration
par.m = m;
par.n = n;
par.max_iter = 100;
par.max_time = 1e6;
par.tol = 1e-3;

% intialize W and H
% W = rand(m,k);
% H = rand(k,n);
W = rand(m,k);
H = rand(k,n);

btk = 0;

acc = 0; 

for t = 1: n/batch_size
    % draw a data sample v_t from P
    vt = V(:, (t-1)*batch_size+1:t*batch_size);
    
    % Learn the coefficient vector h_t per algorithm 2
    ht = rand(k,1);
    for e = 1:1
    ht = learning_h_t(ht, W, vt, btk, 100);
    
    % Update the basis matrix from W_t-1 to W_t
%     for it = 1:1
%     for i = 1:size(W, 1)
%         for a = size(W, 2)
%             temp = W * ht;
%             top = sum (ht(a,:) .* vt(i,:) ./ temp(i,:));
%             W(i,a) = W(i,a) * ( top / sum(ht(a,:), 2));
%         end
%     end
%     end
    
    temp2 = (vt * ht') ./ (W * ht * ht');
    W = W .* temp2;
    W(W<0) = 0;
    W(W>1) = 1;
    end
    
%     disp(t);
%     disp(sum(sum(abs(vt - W * ht) > .5)) / 784 / batch_size * 100);
    
end

% I = find(isnan(W) | W < 1e-5);
% W(I) = 0;
H = pinv(W) * V;

end

function ht = learning_h_t(ht, Wt1, vt, btk, g)
% this is corresponding to algorithm 2 of the paper
% <Inputs>
%       ht: initial coefficient vector h_t_0
%       Wt1: basis matrix W_(t-1)
%       vt: data sample
%       btk: step size beta_t_k
%       g: maximum number of iterations gama
%<Outputs>
%       ht: final coefficient vector h_t := h_t_gama

% Try multiplicative update rule proposed in Lee & Seung's "Algorithms for Non-negative Matrix Factorization"
% as a good compromise of between speed and ease of implementation.
% Here we can ignore btk for now.

for k = 1:g
%     for u = 1:size(ht, 2)
%         for a = 1:size(ht,1)
%             temp = Wt1 * ht;
%             top = sum (Wt1(:,a) .* vt(:,u) ./ temp(:,u));
%             ht(a,u) = ht(a,u) * (top / sum(Wt1(:,a)));
%         end
%     end
    temp1 = (Wt1' * vt) ./ (Wt1' * Wt1 * ht);
    ht = ht .* temp1;
    ht(ht<0) = 0;
    ht(ht>1) = 1;
end
end













