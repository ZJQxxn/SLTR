%% TODO: split 120 samples into a training set and a test set
addpath('fMRI_data')
addpath('tensor_toolbox/')

load('P1_invertX_mat.mat')
load('P1_X_mat.mat')
load('P1_Y_mat.mat')
format short g
p = size(invertX);
M = length(p)-1;
N = p(end);

% split into training and testing set
splitPoint = floor(N*0.8);
indices = randperm(N);
trainIndices = indices(1:splitPoint);
testIndices = indices(splitPoint+1:end);
trainY = Y(trainIndices,:);
trainX = X(trainIndices, :,:,:);
trainInvertX = invertX(:,:,:, trainIndices);
testY = Y(testIndices,:);
testX = X(testIndices, :,:,:);
testInvertX = invertX(:,:,:, testIndices);

%% Prox_Remurs (0, 0, 0.25)
% parameter settings
disp('===== Prox_Remurs =====')
rho = 0.8; % learning rate
minDiff=1e-4;
maxIter=1000;
matrixY=reshape(trainY.data,[splitPoint 1]);
cvTau = 1e-2;
cvLambda = 1e-2;
cvEpsilon = 0.3;
% main step
tic
[estimatedW, errSeq] = Prox_Remurs(double(trainInvertX), matrixY, cvTau, cvLambda, cvEpsilon, rho, maxIter, minDiff);
t = toc;
fprintf('Elapsed time is %.4f sec\n',t)
% ROC and AUC
predY = ttt(tensor(testX), tensor(estimatedW), 2:M+1, 1:M); 
predY = reshape(double(predY),[N-splitPoint 1]);
[rocX, rocY, rocT, AUC] = perfcurve(double(testY), sigmoid(predY), 1);
fprintf('The AUC value of Prox_Remurs is %f\n', AUC);
% save the ROC curve
Prox_ROC.X = rocX;
Prox_ROC.Y = rocY;
Prox_ROC.T = rocT;
Prox_ROC.AUC = AUC;
save('Prox_ROC1.mat', 'Prox_ROC')