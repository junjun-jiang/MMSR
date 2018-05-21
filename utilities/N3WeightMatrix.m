%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% construct the nearest neighbor non-negtive weight matrix
function [WW] = N3WeightMatrix(X,K)

[D,N] = size(X);

% STEP1: COMPUTE PAIRWISE DISTANCES & FIND NEIGHBORS 
X2 = sum(X.^2,1);
distance = repmat(X2,N,1)+repmat(X2',1,N)-2*X'*X;

[sorted,index] = sort(distance);
neighborhood = index(2:(K+1),:);

% STEP2: SOLVE FOR RECONSTRUCTION WEIGHTS
W = zeros(K,N);
WW = zeros(N,N);
% for ii = 1:N
%    W(:,ii) = lsqnonneg(X(:,neighborhood(:,ii)),X(:,ii)); 
%     WW(neighborhood(:,ii),ii) = W(:,ii)';
% end
% for ii = 1:N
%    W(:,ii) = lsqnonneg(X,X(:,ii)); 
%     WW(neighborhood(:,ii),ii) = W(:,ii)';
% end


tol=1e-3; 
for ii=1:N
   z = X(:,neighborhood(:,ii))-repmat(X(:,ii),1,K); % shift ith pt to origin
   C = z'*z;                                        % local covariance
   C = C + eye(K,K)*tol*trace(C);                   % regularlization (K>D)
%    W(:,ii) = C\ones(K,1);                           % solve Cw=1
    W(:,ii) = pinv(C'*C)*C'*ones(K,1);
   W(:,ii) = W(:,ii)/sum(W(:,ii));                  % enforce sum(w)=1
   WW(neighborhood(:,ii),ii) = W(:,ii)';
end;

% WW = (WW+WW')/2;
% WW = WW + eye(N,N);



