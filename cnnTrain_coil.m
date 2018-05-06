%% Convolution Neural Network Exercise

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started in building a single.
%  layer convolutional nerual network. In this exercise, you will only
%  need to modify cnnCost.m and cnnminFuncSGD.m. You will not need to 
%  modify this file.

%%======================================================================
%% STEP 0: Initialize Parameters and Load Data
%  Here we initialize some parameters used for the exercise.

clear all
close all
clc
% Configuration
imageDim = 128;
numClasses = 100;  % Number of classes (MNIST images fall into 10 classes)
filterDim = 9;    % Filter size for conv layer
numFilters = 20;   % Number of filters for conv layer
poolDim = 2;      % Pooling dimension, (should divide imageDim-filterDim+1)

% Load MNIST Train

load('im_train_coil');
images = im_train;
load('label_train_coil');
labels = label_train;

% images_t = zeros(imageDim, imageDim, 100*5);
% labels_t = zeros(100*5,1);
% 
% for i = 1:5
%     x = find(labels==i);
%     for j = 1:100
%         y = images(:,:,x(j));
%         images_t(:, :, (i-1)*100+j) = y;
%     end
% end
% for i = 1:5
%     for j = 1:100
%         labels_t((i-1)*100+j) = i;
%     end
% end

% Initialize Parameters
theta = cnnInitParams(imageDim,filterDim,numFilters,poolDim,numClasses);

%%======================================================================
%% STEP 1: Implement convNet Objective
%  Implement the function cnnCost.m.

%%======================================================================
%% STEP 2: Gradient Check
%  Use the file computeNumericalGradient.m to check the gradient
%  calculation for your cnnCost.m function.  You may need to add the
%  appropriate path or copy the file to this directory.

DEBUG=false;  % set this to true to check gradient
if DEBUG
    % To speed up gradient checking, we will use a reduced network and
    % a debugging data set
    db_numFilters = 2;
    db_filterDim = 9;
    db_poolDim = 5;
    db_images = images(:,:,1:10);
    db_labels = labels(1:10);
    db_theta = cnnInitParams(imageDim,db_filterDim,db_numFilters,...
                db_poolDim,numClasses);
    
    [cost grad] = cnnCost(db_theta,db_images,db_labels,numClasses,...
                                db_filterDim,db_numFilters,db_poolDim);
    

    % Check gradients
    numGrad = computeNumericalGradient( @(x) cnnCost(x,db_images,...
                                db_labels,numClasses,db_filterDim,...
                                db_numFilters,db_poolDim), db_theta);
 
    % Use this to visually compare the gradients side by side
    disp([numGrad grad]);
    
    diff = norm(numGrad-grad)/norm(numGrad+grad);
    % Should be small. In our implementation, these values are usually 
    % less than 1e-9.
    disp(diff); 
 
    assert(diff < 1e-9,...
        'Difference too large. Check your gradient computation again');
    
end;

%%======================================================================
%% STEP 3: Learn Parameters
%  Implement minFuncSGD.m, then train the model.

options.epochs = 8;
options.minibatch = 256;
options.alpha = 1e-1;
options.momentum = .95;

opttheta = minFuncSGD(@(x,y,z) cnnCost(x,y,z,numClasses,filterDim,...
                      numFilters,poolDim),theta,images,labels,options);


%%======================================================================
%% STEP 4: Test
%  Test the performance of the trained model using the MNIST test set. Your
%  accuracy should be above 97% after 3 epochs of training

load('im_test_coil');
testImages = im_test;
load('label_test_coil');
testLabels = label_test;

% testImages_t = zeros(imageDim, imageDim, 10*5);
% testLabels_t = zeros(10*5,1);
% 
% for i = 1:5
%     t = find(testLabels==i);
%     for j = 1:10
%         r = testImages(:,:,t(j));
%         testImages_t(:, :, (i-1)*10+j) = r;
%     end
% end
% for i = 1:5
%     for j = 1:10
%         testLabels_t((i-1)*10+j) = i;
%     end
% end
% 
% [~,cost,preds]=cnnCost(opttheta,testImages_t,testLabels_t,numClasses,...
%                 filterDim,numFilters,poolDim,true);
% 
% acc = sum(preds==testLabels_t)/length(preds);
% 
[~,cost,preds]=cnnCost(opttheta,testImages,testLabels,numClasses,...
                filterDim,numFilters,poolDim,true);

acc = sum(preds==testLabels)/length(preds);


% Accuracy should be around 97.4% after 3 epochs
fprintf('Accuracy is %f\n',acc);