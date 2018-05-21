function [hIm, ww] = L1SR_Graph(lIm, zooming, patch_size, overlap, Dh, Dl, lambda, regres, K1, Gam,method_flag)
% Use sparse representation as the prior for image super-resolution
% Usage
%       [hIm] = L1SR(lIm, zooming, patch_size, overlap, Dh, Dl, lambda)
% 
% Inputs
%   -lIm:           low resolution input image, single channel, e.g.
%   illuminance
%   -zooming:       zooming factor, e.g. 3
%   -patch_size:    patch size for the low resolution image
%   -overlap:       overlap among patches, e.g. 1
%   -Dh:            dictionary for the high resolution patches
%   -Dl:            dictionary for the low resolution patches
%   -regres:       'L1' use the sparse representation directly to high
%                   resolution dictionary;
%                   'L2' use the supports found by sparse representation
%                   and apply least square regression coefficients to high
%                   resolution dictionary.
% Ouputs
%   -hIm:           the recovered image, single channel
%
% Written by Jianchao Yang @ IFP UIUC
% April, 2009
% Webpage: http://www.ifp.illinois.edu/~jyang29/
% For any questions, please email me by jyang29@uiuc.edu
%
% Reference
% Jianchao Yang, John Wright, Thomas Huang and Yi Ma. Image superresolution
% as sparse representation of raw image patches. IEEE Computer Society
% Conference on Computer Vision and Pattern Recognition (CVPR), 2008. 
%

[lhg, lwd] = size(lIm);
hhg = lhg*zooming;
hwd = lwd*zooming;

mIm = imresize(lIm, 2,'bicubic');
[mhg, mwd] = size(mIm);
hpatch_size = patch_size*zooming;
mpatch_size = patch_size*2;

% extract gradient feature from lIm
hf1 = [-1,0,1];
vf1 = [-1,0,1]';
hf2 = [1,0,-2,0,1];
vf2 = [1,0,-2,0,1]';

lImG11 = conv2(mIm,hf1,'same');
lImG12 = conv2(mIm,vf1,'same');
lImG21 = conv2(mIm,hf2,'same');
lImG22 = conv2(mIm,vf2,'same');

lImfea(:,:,1) = lImG11;
lImfea(:,:,2) = lImG12;
lImfea(:,:,3) = lImG21;
lImfea(:,:,4) = lImG22;

lgridx = 2:patch_size-overlap:lwd-patch_size;
lgridx = [lgridx, lwd-patch_size];
lgridy = 2:patch_size-overlap:lhg-patch_size;
lgridy = [lgridy, lhg-patch_size];

mgridx = (lgridx - 1)*2 + 1;
mgridy = (lgridy - 1)*2 + 1;

% using linear programming to find sparse solution
bhIm = imresize(lIm, 3, 'bicubic');
hIm = zeros([hhg, hwd]);
nrml_mat = zeros([hhg, hwd]);

hgridx = (lgridx-1)*zooming + 1;
hgridy = (lgridy-1)*zooming + 1;

disp('Processing the patches sequentially...');
count = 0;

% loop to recover each patch
for xx = 1:length(mgridx),
    for yy = 1:length(mgridy),
        
        mcolx = mgridx(xx);
        mrowy = mgridy(yy);
        
        count = count + 1;
        if ~mod(count, 5000),
            fprintf('.\n');
        else
            if mod(count, 50) ==0
                fprintf('.');
            end
        end;
        
        mpatch = mIm(mrowy:mrowy+mpatch_size-1, mcolx:mcolx+mpatch_size-1);
        mmean = mean(mpatch(:));
        
        mpatchfea = lImfea(mrowy:mrowy+mpatch_size-1, mcolx:mcolx+mpatch_size-1, :);
        mpatchfea = mpatchfea(:);
        
        mnorm = sqrt(sum(mpatchfea.^2));
        
        if mnorm > 1,
            y = mpatchfea./mnorm;
        else
            y = mpatchfea;
        end;
        
%         w = SolveLasso(Dl, y, size(Dl, 2), 'nnlasso', [], 0.1);
%         w = feature_sign(Dl, y, lambda);

        if strcmp('ScSR',method_flag)
%             w = SolveLasso(Dl, y, size(Dl, 2), 'nnlasso', [], 0.1);
            w = feature_sign(Dl, y, 0.1);
            if isempty(w),
                w = zeros(size(Dl, 2), 1);
            end;
        end
       
        if strcmp('MMSR',method_flag)
%             w = SolveLasso(Dl, y, size(Dl, 2), 'nnlasso', [], 0.1);
            w = feature_sign(Dl, y, 0.1);
            if isempty(w),
                w = zeros(size(Dl, 2), 1);
            end;
        end

        if strcmp('LLR',method_flag)
            w = SolveLSR(y,Dl,K1);
            if isempty(w),
                w = zeros(size(Dl, 2), 1);
            end;
        end

        if strcmp('NESR',method_flag)              

        end    
        

        switch regres,
            case 'L1'
                if mnorm > 1,
                    hpatch = Dh*w*mnorm;
                else
                    hpatch = Dh*w;
                end;
            case 'L2'
             
               
                if strcmp('MMSR',method_flag)
                    idx = find(w);
                    lsups = Dl(:, idx);
                    hsups = Dh(:, idx);
                    if size(idx,1)>K1
                       K =  K1;
                    else
                        if size(idx,1)>1
                            K = size(idx,1)-1;
                        else
                            K = 1;
                        end
                    end  

                  A_P =  CalulateProjMatrix(lsups,hsups,lambda,K,Gam); % obtain the projection matrix A
                  hpatch = A_P'*mpatchfea;    
                end
                
                if strcmp('LLR',method_flag)
                    idx = find(w);
                    lsups = Dl(:, idx);
                    hsups = Dh(:, idx);
                    w = inv(lsups'*lsups+Gam*eye(size(lsups,2),size(lsups,2)))*lsups'*mpatchfea;
                    hpatch = hsups*w;                  
                end
                
                if strcmp('NESR',method_flag)         
                    [neighborhood w] = NN(y,Dl,5);
                    Dhh = Dh(:,neighborhood);
                    hpatch = Dhh*w;                     
                end
                
                if strcmp('NEIternation',method_flag)         
                    [neighborhood w] = NNIternation(y,Dl,Dh,K1);
                    Dhh = Dh(:,neighborhood);
                    hpatch = Dhh*w;                     
                end
                
                if strcmp('NEGraph',method_flag)%         
                    [neighborhood w] = NNGraph(y,Dl,Dh,30,Gam);
                    Dhh = Dh(:,neighborhood);
                    hpatch = Dhh*w;                     
                end
                
                if strcmp('ScSR',method_flag)         
                    idx = find(w);
                    lsups = Dl(:, idx);
                    hsups = Dh(:, idx);
                    w = inv(lsups'*lsups)*lsups'*mpatchfea;
                    hpatch = hsups*w;                 
                end
                
            otherwise
                error('Unknown fitting!');
        end;
      
        hpatch = reshape(hpatch, [hpatch_size, hpatch_size]);
        hpatch = hpatch + mmean;
        
        hcolx = hgridx(xx);
        hrowy = hgridy(yy);
        
        hIm(hrowy:hrowy+hpatch_size-1, hcolx:hcolx+hpatch_size-1)...
            = hIm(hrowy:hrowy+hpatch_size-1, hcolx:hcolx+hpatch_size-1) + hpatch;
        nrml_mat(hrowy:hrowy+hpatch_size-1, hcolx:hcolx+hpatch_size-1)...
            = nrml_mat(hrowy:hrowy+hpatch_size-1, hcolx:hcolx+hpatch_size-1) + 1;
    end;
end;

fprintf('done!\n');

% fill the empty
hIm(1:3, :) = bhIm(1:3, :);
hIm(:, 1:3) = bhIm(:, 1:3);

hIm(end-2:end, :) = bhIm(end-2:end, :);
hIm(:, end-2:end) = bhIm(:, end-2:end);

nrml_mat(nrml_mat < 1) = 1;
hIm = hIm./nrml_mat;
% hIm = uint8(hIm);

