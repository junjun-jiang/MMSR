
function [neighborhood W] = NNIternation(x,X,XF,K)

% STEP1: COMPUTE PAIRWISE DISTANCES & FIND NEIGHBORS
x   = double(x);
n2 = dist2(x', X');
[value index] = sort(n2);
neighborhood = index(1:K);

% STEP2: SOLVE FOR RECONSTRUCTION WEIGHTS
W = solve_weight(X(:,neighborhood),x,K);
y_pre = XF(:,neighborhood)*W;

% y_pre = y;

maxiter = 10;
for i=1:maxiter
    y = y_pre;    
    n2 = dist2(y', XF');
    [value index] = sort(n2);
    neighborhood = index(1:K);
%     W = solve_weight([X(:,neighborhood);.001*XF(:,neighborhood)],[x;.001*y],K);
    W = solve_weight(X(:,neighborhood),x,K);
    y_pre = XF(:,neighborhood)*W;
%     (y-y_pre)'*(y-y_pre)
end

function W = solve_weight(X,x,K);
tol=1e-6;
z = X-repmat(x,1,K); % shift ith pt to origin
C = z'*z;                                        % local covariance

if trace(C)==0
    C = C + eye(K,K)*tol;                   % regularlization
else
    C = C + eye(K,K)*tol*trace(C);         % regularlization (K>D)
end
                
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
