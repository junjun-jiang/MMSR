
clear all;
clc;
close all;

addpath('Solver');
addpath('Sparse coding');
addpath('.\utilities')
% =====================================================================
% specify the parameter settings

patch_size = 3; % patch size for the low resolution input image
overlap = 1; % overlap between adjacent patches
lambda = 0.1; % sparsity parameter
zooming = 3; % zooming factor, if you change this, the dictionary needs to be retrained.

tr_dir = 'Data/training'; % path for training images
skip_smp_training = true; % sample training patches
skip_dictionary_training = true; % train the coupled dictionary
num_patch = 2500; % number of patches to sample as the dictionary
codebook_size = 1024; % size of the dictionary

regres = 'L2'; % 'L1' or 'L2', use the sparse representation directly, or use the supports for L2 regression

% =====================================================================
% Process the test image 
% girl : 4 2 4 33.4025
% lambdas = [0.0005 0.001 0.01 0.1 0.5 1 2 4 8 12];
% Ks = [5 15 30 45 60];
% Gams = [0.5 0.8 1 2 4 8 15 25 40];

method_flag = 'MMSR';%when choose 'ScSR', the regre term is 'L1'
% 10 15 0.15 28.7357
lambdas = [10];%图约束项的参数
Ks = [150];
Gams = [0.3];%对应文章中的alphi,平滑作用


% =====================================================================
% training coupled dictionaries for super-resolution

if ~skip_smp_training,
    disp('Sampling image patches...');
    [Xh, Xl] = rnd_smp_dictionary(tr_dir, patch_size, zooming, num_patch);
    save('Data/Dictionary/smp_patches.mat', 'Xh', 'Xl');
    skip_dictionary_training = false;
    Dh = Xh;
    Dl = Xl;
end;

if ~skip_dictionary_training,
    load('Data/Dictionary/smp_patches.mat');
    [Dh, Dl] = coupled_dic_train(Xh, Xl, codebook_size, lambda);
    save('Data/Dictionary/Dictionary.mat', 'Dh', 'Dl');
else
    load('Data/Dictionary/Dictionary.mat');
end;

if strcmp(method_flag,'NESR')==1 || strcmp(method_flag,'LLR')==1
    [Dh, Dl] = rnd_smp_dictionary(tr_dir, patch_size, zooming, num_patch);
    [Dh, Dl] = patch_pruning(Dh, Dl, 30);
    Dh = NormalizeFea(Dh);
    Dl = NormalizeFea(Dl);
end


% Names = {'baboon','barbara','bike','Butterfly','coastguard','comic','flower','flowers','foreman','girl','hat','house','leaves',...
%     'lenna','lighthouse','Parrots','Parthenon','pepper','plants','raccoon','zebra'};
% Names = {'barbara','Butterfly','coastguard','flower','foreman','house','leaves',...
%     'lenna','lighthouse','plants','zebra'};
% Names = {'kodim01','kodim02','kodim03','kodim04','kodim05','kodim06','kodim07','kodim08','kodim09','kodim10','kodim11','kodim12','kodim13',...
%     'kodim14','kodim15','kodim16','kodim17','kodim18','kodim19','kodim20','kodim21','kodim22','kodim23','kodim24'};
Names = {'barbara','foreman','house','lenna','zebra'};

PSNR_ALL = zeros(size(Names,2),size(lambdas,2),size(Ks,2),size(Gams,2));
PSNR_ALL_BP = zeros(size(Names,2),size(lambdas,2),size(Ks,2),size(Gams,2));
PSNR_ALL_Bicubic = zeros(size(Names,2),size(lambdas,2),size(Ks,2),size(Gams,2));
RMSE_ALL = zeros(size(Names,2),size(lambdas,2),size(Ks,2),size(Gams,2));
RMSE_ALL_BP = zeros(size(Names,2),size(lambdas,2),size(Ks,2),size(Gams,2));
RMSE_ALL_Bicubic = zeros(size(Names,2),size(lambdas,2),size(Ks,2),size(Gams,2));
SSIM_ALL = zeros(size(Names,2),size(lambdas,2),size(Ks,2),size(Gams,2));
SSIM_ALL_BP = zeros(size(Names,2),size(lambdas,2),size(Ks,2),size(Gams,2));
SSIM_ALL_Bicubic = zeros(size(Names,2),size(lambdas,2),size(Ks,2),size(Gams,2));

for name_flag = 1:size(Names,2);
    for ilambda = 1:size(lambdas,2)
        for iK = 1:size(Ks,2)
            for iGam = 1:size(Gams,2)
                lambda = lambdas(ilambda);
                K = Ks(iK);
                Gam = Gams(iGam);
            %     ReconIm = [];
                % read ground truth image
                fname = strcat('.\Data\test\',char(Names(name_flag)), '.bmp');
                testIm = imread(fname); % testIm is a high resolution image, we downsample it and do super-resolution

                if rem(size(testIm,1),zooming) ~=0,
                    nrow = floor(size(testIm,1)/zooming)*zooming;
                    testIm = testIm(1:nrow,:,:);
                end;
                if rem(size(testIm,2),zooming) ~=0,
                    ncol = floor(size(testIm,2)/zooming)*zooming;
                    testIm = testIm(:,1:ncol,:);
                end;

%                 imwrite(testIm, ['Data/Test/Results/high_' char(Names(name_flag)) '.bmp'], 'BMP');

                lowIm = imresize(testIm,1/zooming, 'bicubic');
%                 imwrite(lowIm,['Data/Test/Results/low_' char(Names(name_flag)) '.bmp'],'BMP');

                interpIm = imresize(lowIm,zooming,'bicubic');
%                 imwrite(uint8(interpIm),['Data/Test/Results/bb_' char(Names(name_flag)) '.bmp'],'BMP');

                % work with the illuminance domain only
                lowIm2 = rgb2ycbcr(lowIm);
                lImy = double(lowIm2(:,:,1));

                % bicubic interpolation for the other two channels
                interpIm2 = rgb2ycbcr(interpIm);
                hImcb = interpIm2(:,:,2);
                hImcr = interpIm2(:,:,3);

                % ======================================================================
                % Super-resolution using sparse representation

                disp('Start superresolution...');

                [hImy] = L1SR_Graph(lImy, zooming, patch_size, overlap, Dh, Dl, lambda, regres, K, Gam,method_flag);

                ReconIm = uint8(zeros(size(testIm)));
                ReconIm(:,:,1) = uint8(hImy);    ReconIm(:,:,2) = hImcb;    ReconIm(:,:,3) = hImcr;

                nnIm = imresize(lowIm, zooming, 'nearest');
                ReconIm = ycbcr2rgb(ReconIm);
                imwrite(uint8(ReconIm),['Data/Test/Results/',char(method_flag) '_' char(Names(name_flag)) num2str(Ks(iK)) '.bmp'],'BMP');


                % compute PSNR for the illuminance channel
                bb_rmse = compute_rmse(testIm, interpIm);    
                sp_rmse = compute_rmse(testIm, ReconIm);
                bb_psnr = 20*log10(255/bb_rmse);    
                sp_psnr = 20*log10(255/sp_rmse);
                bb_ssim = ssim(testIm,interpIm);    
                sp_ssim = ssim(testIm, ReconIm);
                
                
                %figure,imshow(uint8(testIm));title('HR');
                %figure,imshow(uint8(interpIm));title('Bicubic');
                %figure,imshow(uint8(ReconIm));title('SR');
                
                fprintf(['PSNR for ',char(Names(name_flag)),' of ',char(method_flag),': %f dB\n'], sp_psnr);
                fprintf('PSNR for Bicubic Interpolation: %f dB\n', bb_psnr);
%                 fprintf('RMSE for Bicubic Interpolation: %f dB\n', bb_rmse);
%                 fprintf('PSNR for Sparse Representation Recovery: %f dB\n', sp_psnr);
%                 fprintf('RMSE for Sparse Representation Recovery: %f dB\n', sp_rmse);

                SSIM_ALL(name_flag,ilambda,iK,iGam) = sp_ssim;   
                PSNR_ALL(name_flag,ilambda,iK,iGam) = sp_psnr;
                RMSE_ALL(name_flag,ilambda,iK,iGam) = sp_rmse;    
                SSIM_ALL_Bicubic(name_flag,ilambda,iK,iGam) = bb_ssim;
                PSNR_ALL_Bicubic(name_flag,ilambda,iK,iGam) = bb_psnr;    
                RMSE_ALL_Bicubic(name_flag,ilambda,iK,iGam) = bb_rmse;

                [hImy] = backprojection(hImy, lImy, 20);
                ReconIm(:,:,1) = uint8(hImy);    ReconIm(:,:,2) = hImcb;    ReconIm(:,:,3) = hImcr;

                ReconIm = ycbcr2rgb(ReconIm);
                imwrite(uint8(ReconIm),['Data/Test/Results/' char(method_flag) '_' char(Names(name_flag)) num2str(Ks(iK)) 'BP.bmp'],'BMP');

%                 %figure,imshow(uint8(ReconIm));title('SR+BP');
                % compute PSNR for the illuminance channel
                sp_rmse = compute_rmse(testIm, ReconIm);
                sp_psnr = 20*log10(255/sp_rmse);
                sp_ssim = ssim(testIm, ReconIm);

                fprintf(['PSNR for ',char(Names(name_flag)),' of ',char(method_flag),'after BP=20: %f dB\n'], sp_psnr);
                
                [hImy] = backprojection(hImy, lImy, 20);
                ReconIm(:,:,1) = uint8(hImy);    ReconIm(:,:,2) = hImcb;    ReconIm(:,:,3) = hImcr;

                ReconIm = ycbcr2rgb(ReconIm);
                imwrite(uint8(ReconIm),['Data/Test/Results/' char(method_flag) '_' char(Names(name_flag)) num2str(Ks(iK)) 'BP.bmp'],'BMP');
                %figure,imshow(uint8(ReconIm));title('SR');

                % compute PSNR for the illuminance channel
                sp_rmse = compute_rmse(testIm, ReconIm);
                sp_psnr = 20*log10(255/sp_rmse);
                sp_ssim = ssim(testIm, ReconIm);

                fprintf(['PSNR for ',char(Names(name_flag)),' of ',char(method_flag),'after BP=50: %f dB\n'], sp_psnr);              
                
                
                [hImy] = backprojection(imresize(lImy,zooming,'bicubic'), lImy, 50);
                ReconIm(:,:,1) = uint8(hImy);    ReconIm(:,:,2) = hImcb;    ReconIm(:,:,3) = hImcr;
                ReconIm = ycbcr2rgb(ReconIm);
                %figure,imshow(uint8(ReconIm));title('Bicubic+BP');
                fprintf(['PSNR for ',char(Names(name_flag)),' of ',char(method_flag),'after BP=20: %f dB\n'], 20*log10(255/compute_rmse(testIm, ReconIm)));
                
%                 fprintf('RMSE for Sparse Representation Recovery: %f dB\n', sp_rmse);
                SSIM_ALL_BP(name_flag,ilambda,iK,iGam) = sp_ssim;
                PSNR_ALL_BP(name_flag,ilambda,iK,iGam) = sp_psnr;
                RMSE_ALL_BP(name_flag,ilambda,iK,iGam) = sp_rmse;
            end
        end        
    end
end

PSNRR = [PSNR_ALL PSNR_ALL_BP PSNR_ALL_Bicubic RMSE_ALL RMSE_ALL_BP RMSE_ALL_Bicubic SSIM_ALL SSIM_ALL_BP SSIM_ALL_Bicubic];
xlswrite([char(method_flag) 'K=20 Gam=1.xls'],PSNRR);