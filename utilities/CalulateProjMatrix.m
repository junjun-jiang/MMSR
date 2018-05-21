


function A_P = CalulateProjMatrix(Dl,Dh,lambda,K,Gam);
%%% Dl : LR training patches
%%% Dh : HR training patches

% normalize the dictionarDh
norm_Dl = sqrt(sum(Dl.^2, 1)); 
Dl = Dl./repmat(norm_Dl, size(Dl, 1), 1);

norm_Dh = sqrt(sum(Dh.^2, 1)); 
Dh = Dh./repmat(norm_Dh, size(Dh, 1), 1);

[feaNum TrainNum] = size(Dl);
% options = [];
% options.Metric = 'Euclidean';
% options.NeighborMode = 'KNN';
% options.k = K;
% [W, elapse] = constructW(Dh',options);
if K ==1
    W = 1;
else
%     [W] = N3WeightMatrix(Dh,K);
    [W] = SparseWeightMatrix(Dh);
end

D = diag(sum(W));
I = eye(TrainNum,TrainNum);
G = Dl*(I-W)*(I-W)'*Dl';

A_P = inv(Dl*Dl' + lambda*G + Gam*eye(feaNum,feaNum))*Dl*Dh';

% U = Dl*Dl' + lambda*G + Gam*eye(feaNum,feaNum);
% V = Dl*Dh';
% for i = 1:size(V,2)
%     A_P(:,i) = lsqnonneg(U,V(:,i));
% end
