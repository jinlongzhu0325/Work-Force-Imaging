% This program implements a robust, matrix-based, regularized,
% non-iterative solution for the cencept of Work-Force imaging.
% The program takes approximately 9 seconds to run on a 10 megapixel image
% on a desktop computer with a i7-4770 3.4GHz processor.
%
% If you use this code in your research, please cite both the original
% paper in which the method was presented and the code's figshare page:
%
% [1] J. Zhu, R. Zhou, Q. Ding, K. Toussaint, Jr., P. Braun, and L. L. Goddard, 
% "Conservation in Non-Conservative Optical Force: Work-Force Imaging for Nanoscale 
% Characterization" Phys. Rev. Lett., submitted to publication.
%
% [2] J. Zhu, R. Zhou, L. Zhang, B. Ge, C. Luo, and L. L. Goddard, 
% "Regularized pseudo-phase imaging for inspecting and sensing nanoscale features
% ," Opt.Express 27, 6719-6733(2019).
%
% [3] J. Zhu and L. L. Goddard, figshare (2017) [retrieved {enter date}],
% https://figshare.com/articles/Robust_TIE_Solver_m/5504086.
%
% ## How to use this program ##
% Each image must be a 2d array of doubles. Rectangular input images will
% automatically be cropped into a square image of the central region.
% This function has the following input and output arguments:
%
% ## Inputs ##
% I0_pos - intensity image at a distance dz above the best focus plane
% I0 - intensity image at the best focus plane
% I0_neg - intensity image at a distance dz below the best focus plane
% dz - the distance in z (in microns) between each of the three images
% lambda - central wavelength (in microns) of the white light source
% reg_factor - a small parameter (usually 0.00001) used to regularize
%
% ## Outputs ##
% work - computed output work image
% force - computed output force image
%
% Acknowledgment:
% Images of red blood cells (courtesy of Baoliang Ge, MIT) are used as an
% example for running the program.
%
% Creative Commons License:
% Attribution-NonCommercial-NoDerivatives 4.0 International
% Copyright by Jinlong Zhu and Lynford L. Goddard
% Photonic Systems Lab, University of Illinois at Urbana-Champaign.
% Version 1.3, 05-08-2019.
%
% ####################################################################
clear; clc;

%% ## Specify experimental conditions ##
um_per_pixel = 6.45/106.6667;   % size of an pixel in sample plane (in microns). In the DPM, the pixel size is 6.45um and the magnification is 106.66
dz = 0.5;                       % translation distance of the sample (in microns)
Neumann = true;               % Neumann (true) or Dirichlet (false) boundaries
lambda = 0.633;               % vacuum wavelength (in microns)
ref_index = 1.33;                % index of ambient environment
reg_factor = 5e-5;            % regularization factor (1e-6) for RBC


%% ## Specify the systematic parameters ##
use_gpu = 'false';
use_single = 'false';


%% Calculate quantities that don't change in a given experiment
dx2 = um_per_pixel^2;    % dx^2 = dy^2 in microns (used for 2D Laplacian)
k0 = 2*pi*ref_index/lambda;  % wave vector

% read a sample image for getting its numbers of columns and rows
% image_name = strcat('1','.tiff');
I0 = im2double(rgb2gray(imread('Image_-0.5.tif')));
% imageData = read(t);
% I0 = im2double(imageData);    
rows = length(I0(:,1));

columns = length(I0(1,:));
n = min(rows, columns);      % n is used to crop a rectangular image into a square one


%% ######## Step 0: Initialize matrices (one time calculation) #########

% Step 0a: define T matrix
r = 0*(1:n); r(1) = -2; r(2) = 1; T = toeplitz(r);
Z = zeros(n);
if strcmp(use_single, 'true')
    T = single(T);
    Z = single(Z);
end
if Neumann
    T(1,1) = -1; T(n,n) = -1;
end
% Step 0b: eigenvalue decomposition of T (eigenvectors: Q, eigenvalues: S)
[Q,S] = eig(T);
% Step 0c: T is Hermitian so Q is unitary: Q inverse = Q conj transpose
Qinv = Q'; 
% Step 0d: compute elements of Z (denominator of Eq. 7 with regularization)
% We will pre-calculate Z and use it to find K according to: K = H ./ Z
for i = 1:n
    for j = 1:n
        Z(i,j) = 1/(S(i,i)+S(j,j)) * ...
            ((S(i,i)+S(j,j))^2/((S(i,i)+S(j,j))^2+reg_factor)); %
    end
end

if strcmp(use_gpu, 'true')
    gd = gpuDevice();
    Q = gpuArray(Q);
    Qinv = gpuArray(Qinv);
    Z = gpuArray(Z);
end

% Step 0e: compute the crop window for future images
if rows<columns
    crop_cols = 1+(columns-rows)/2:(columns+rows)/2;
    crop_rows = 1:rows;
else
    crop_rows = 1+(rows-columns)/2:(rows+columns)/2;
    crop_cols = 1:columns;
end


%% ######### Solve the Equation #########
% read the .tiff images from a specifed folder 
% change the current folder)
% folder_name1 = 'H:\Photonic System Laboratoty\Projects\FOCI Project\Phase 2\TIE Test\405nm_Broadband_No_Polarizater\Position2';
% folder_name2 = 'delta_x_0nm';
% cd([folder_name1, ' (',folder_name2,')']);
% files = dir('*.tiff');                                            % specify the type the files
% m     = size(files,1);                                            % number of files
rhs_const = -k0 * dx2 / (2*dz);                                   % scaling constant for RHS of TIE
m = 1;
% m = 3 : 1: 19;

phase = zeros(1024,1024,length(m));
work = zeros(1024,1024,length(m));
I_orig = zeros(1024,1024,length(m));

I0     = im2double(rgb2gray(imread('Image_0.tif')));
I0_pos = im2double(rgb2gray(imread('Image_0.5.tif')));
I0_neg = im2double(rgb2gray(imread('Image_-0.5.tif')));

% i = 237; % specify a given focal plane
for j = 1 : 1
    % Step 1: Crop input images to make them square
 
%     t = Tiff(['FIB_4s_Exposure_Single_405nm\Four_Expos_',num2str(18),'.tiff'],'r');
    I0_neg = im2double(rgb2gray(imread('Image_-0.5.tif')));
%     imageData = read(t);
%     I0_neg = im2double(imageData);
%     figure,imagesc(I0_neg),colormap hot
    
    % read the in-focus image
%     t = Tiff(files(i).name,'r');
    I0 = im2double(rgb2gray(imread('Image_0.tif')));
%     imageData = read(t);
%     I0 = im2double(imageData);
    
    % read the positive image
%     t = Tiff(files(i+1).name,'r');
    I0_pos = im2double(rgb2gray(imread('Image_0.5.tif')));
%     imageData = read(t);
%     I0_pos = im2double(imageData);
    
    if strcmp(use_single, 'true')
        I0_pos = single(I0_pos);
        I0 = single(I0);
        I0_neg = single(I0_neg);
    end
    
    if strcmp(use_gpu, 'true')
        I0_crop = gpuArray(I0(crop_rows,crop_cols));
        TIE_RHS = gpuArray(rhs_const * (I0_pos(crop_rows,crop_cols) - ...
        I0_neg(crop_rows,crop_cols)));
    else
        I0_crop = I0(crop_rows,crop_cols);
        TIE_RHS = rhs_const * (I0_pos(crop_rows,crop_cols) - ...
        I0_neg(crop_rows,crop_cols));
    end
    
    % Steps 2-4: Recover the phase. These 5 lines can be put inside a for loop
    % to execute on multiple measurements taken with the same configuration.
    F = Q * ((Qinv * TIE_RHS * Q) .* Z) * Qinv;                                 % Step 2
    work(:,:,j) = F;
    w_C = -(I0_pos(crop_rows,crop_cols) - I0_neg(crop_rows,crop_cols)) / (2*dz);
    
    [grad1x_F,grad1y_F] = gradient(F);                                          % Step 3a
    [grad2x_phi,~] = gradient(grad1x_F./I0_crop);                               % Step 3b
    [~,grad2y_phi] = gradient(grad1y_F./I0_crop);                               % Step 3c
    phase(:,:,j) = Q * ((Qinv * (grad2x_phi + grad2y_phi) * Q) .* Z) * Qinv;  % Step 4
%     wait(gd) % need to uncomment this for proper timing
    

%     cd('H:\Photonic System Laboratoty\Projects\FOCI Project\Phase 2\TIE Test'); 
    % Step 5: Check the agreement of the TIE solution
    [grad1_phase_x,grad1_phase_y] = gradient(phase(:,:,j));
    [grad2_F_x,~] = gradient(I0_crop.*grad1_phase_x);
    [~,grad2_F_y] = gradient(I0_crop.*grad1_phase_y);
    difference = TIE_RHS - (grad2_F_x + grad2_F_y);
    
    I_orig(:,:,j) = I0_crop;  % save the I0_crop to the variable I_orig
    
    clc
    disp(['Total number of calculations is ',num2str(length(m))])
    disp(['Current computing number is ',num2str(j)])
end

%% ########## draw ############
% figure
% subplot(2,3,1)
% imagesc(I0_crop/max(max(I0_crop)))
% axis off
% axis image
% colorbar
% 
% subplot(2,3,2)
% imagesc(w_C/max(max(abs(w_C))))
% axis off
% axis image
% colorbar
% 
% subplot(2,3,3)
% imagesc(work/max(max(abs(work))))
% axis off
% axis image
% colorbar
% 
% subplot(2,3,4)
% imagesc(grad1x_F/max(max(abs(grad1x_F))))
% axis off
% axis image
% colorbar
% 
% subplot(2,3,5)
% imagesc(grad1y_F/max(max(abs(grad1y_F))))
% axis off
% axis image
% colorbar

II = I0_crop/max(max(I0_crop));
wwCC = w_C/max(max(abs(w_C)));
wwKK = work/max(max(abs(work)));
FF = sqrt(abs(grad1x_F).^2 + abs(grad1y_F).^2);
FF = FF / max(max(FF));


figure
subplot(2,2,1)
imagesc(II)
axis off
axis image
colorbar

subplot(2,2,2)
imagesc(wwCC)
axis off
axis image
colorbar

subplot(2,2,3)
imagesc(wwKK)
axis off
axis image
colorbar

subplot(2,2,4)
imagesc(FF)
axis off
axis image
colorbar

