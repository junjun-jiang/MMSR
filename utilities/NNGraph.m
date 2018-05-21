
function [neighborhood W] = NNGraph(x,X,XF,K,tau)



% STEP1: COMPUTE PAIRWISE DISTANCES & FIND NEIGHBORS
x = double(x);
n2 = dist2(x', X');
[value index] = sort(n2);
neighborhood = index(1:K);

% STEP2: CONSTRUCT THE lAPLACIAN MATRIX L FOR HR PATCHES
L = constructLaplacian(XF(:,neighborhood));


% STEP3: SOLVE FOR RECONSTRUCTION WEIGHTS
tol=1e-6;
z = X(:,neighborhood)-repmat(x,1,K); % shift ith pt to origin
C = z'*z;                                        % local covariance
C = C + eye(K,K)*tol*trace(C)+tau*L;                   % regularlization (K>D)
W = C\ones(K,1);                           % solve Cw=1
W = W/sum(W);                  % enforce sum(w)=1



function n2 = dist2(x, c)
%DIST2	Calculates squared distance between two sets of points.
%
%	Description
%	D = DIST2(X, C) takes two matrices of vectors and calculates the
%	squared Euclidean distance between them.  Both matrices must be of
%	the same column dimension.  If X has M rows and N columns, and C has
%	L rows and N columns, then the result has M rows and L columns.  The
%	I, Jth entry is the  squared distance from the Ith row of X to the
%	Jth row of C.
%
%	See also
%	GMMACTIV, KMEANS, RBFFWD
%

%	Copyright (c) Ian T Nabney (1996-2001)

[ndata, dimx] = size(x);
[ncentres, dimc] = size(c);
if dimx ~= dimc
	error('Data dimension does not match dimension of centres')
end

n2 = (ones(ncentres, 1) * sum((x.^2)', 1))' + ...
  ones(ndata, 1) * sum((c.^2)',1) - ...
  2.*(x*(c'));

% Rounding errors occasionally cause negative entries in n2
if any(any(n2<0))
  n2(n2<0) = 0;
end


function L = constructLaplacian(X,tau)


% options.NN = K-1;
% options.GraphDistanceFunction = 'euclidean';
% options. WEIGHTTYPPE = 'heat';
% options.NORMALIZE = 1;
% 
% [L,options] = laplacian(options,X);  

options = [];
options.Metric = 'Euclidean';
options.NeighborMode = 'KNN';
options.k = size(X,2)-1;
options.WeightMode = 'HeatKernel';
df2=dist2(X,X);
options.t = sqrt(mean(df2(:)));
[W, elapse] = constructW(X',options);
    L = diag(sum(W))-W;


% %Normalizing the Graph Laplacian
% D = sum(W(:,:),2); 
% % fprintf(1, 'Normalizing the Graph Laplacian\n');
% D(find(D))=sqrt(1./D(find(D)));
% D=spdiags(D,0,speye(size(W,1)));
% W=D*W*D;
% L=speye(size(W,1))-W;