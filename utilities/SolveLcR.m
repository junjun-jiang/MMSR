function w = SolveLcR(im_l_patch,X,tau)
[nfea nTraining] = size(X);
% represent the LR patch at  position£¨i,j£©using our LcR 
nframe=size(im_l_patch',1);
nbase=size(X',1);
XX = sum(im_l_patch'.*im_l_patch', 2);        
SX = sum(X'.*X', 2);
D  = repmat(XX, 1, nbase)-2*im_l_patch'*X+repmat(SX', nframe, 1); % Calculate the distance between the input LR image patch and the LR training image patches at position£¨i,j£©
% Compute the optimal weight vector  for the input LR image patch  with the LR training image patches at position£¨i,j£©
% z = X' - repmat(im_l_patch', nTraining, 1);         
% C = z*z';                                                
% C = C + tau*diag((D.^0.125)/1e+6)+eye(nTraining,nTraining)*(1e-6)*trace(C);   
% w = C\ones(nTraining,1);  
% w = w/sum(w); 
w = zeros(nTraining,1);
[value index]=sort(D);
w(index(1:20)) = 1;