
function [W] = SparseWeightMatrix(X,tol)
% construct the sparse weight matrix for the input X
 if nargin == 1
     tol = 1e-6;
 end

% X = rand(25,124);

norm_X = sqrt(sum(X.^2, 1)); 
X = X./repmat(norm_X, size(X, 1), 1);
 
% addpath('.\utilities\Optimization');

[nfea,nspl] = size(X);
W = zeros(nspl,nspl);

for i = 1:nspl
    x = X(:,i);
    XX = X;
    XX(:,i) = [];
    w0 = zeros(nspl-1,1);
%     w = l1eq_pd(w0, XX, [], x, 0.1);  
%     w = SolveLasso(XX, x, size(XX, 2), 'lasso', [], 0.05);
    w = feature_sign(XX, x, 0.15);
%     figure,plot(w)
%     w = L1QP_FeatureSign_yang(0.0001,XX'*XX,-XX'*x);
%     figure,plot(w)
    W(1:i-1,i) = w(1:i-1);
    W(i+1:end,i) = w(i:end);    
end