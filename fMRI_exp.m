%% TODO: split 120 samples into a training set and a test set
addpath('fMRI_data/')
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
cvLambda = 1e-3;
cvEpsilon = 0.2;
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
save('Prox_ROC7.mat', 'Prox_ROC')

%% Remurs
% parameter settings
addpath('RemursCode/Code/')
disp('===== Remurs =====')
disp('Start time')

setting = expSet;
epsilon=1e-4;
iter=1000;
matrixY=reshape(trainY.data,[splitPoint 1]);
cvAlpha = 1e-3;
cvBeta = 1e-3;
% main step
tic
[estimatedW, errList] = Remurs(double(trainInvertX), matrixY, cvAlpha, cvBeta, epsilon, iter);
t = toc;
fprintf('Elapsed time is %.4f sec\n',t)
predY = ttt(tensor(testX), tensor(estimatedW), 2:M+1, 1:M);
predY = reshape(double(predY),[N-splitPoint 1]);
[rocX, rocY, rocT, AUC] = perfcurve(double(testY), sigmoid(predY), 1);
fprintf('The AUC value of Remurs is %f\n', AUC);
% save the ROC curve
Remur_ROC.X = rocX;
Remur_ROC.Y = rocY;
Remur_ROC.T = rocT;
Remur_ROC.AUC = AUC;
save('Remur_ROC7.mat', 'Remur_ROC')
disp('Finish time')

%% SURF
% parameter settings
addpath('SURF_code/')
addpath('SURF_code/tensorlab/')
disp('===== SURF =====')
% convet X to vector for Lasso and ENet
vecX = tenmat(trainX, 1);
vecX = vecX.data;

absW = 1e-3;
%cvAlpha = 0.1;
%cvEpsilon = 0.1;
%cvR = 1;
disp('Start cv......')
clock
[cvAlpha, cvEpsilon, cvR] = cv_SURF(double(trainInvertX), vecX, double(trainY), p(1:M),...
    [5e-4 1e-4 5e-3 1e-2 5e-2 1e-1],...
    [5e-4 1e-4 5e-3 1e-2 5e-2 1e-1],...
    [1 2],...
    5, absW);
disp('Finish cv!')
clock
% time cost
totalTime = 0;
tmpY = trainY;
% main procedure
estimatedW = zeros(cvR,prod(p(1:M)));
tic
for r =1:cvR
    [W_r, residual] = MyTrain(double(trainInvertX), vecX, double(tmpY), cvEpsilon, cvEpsilon^2/2, cvAlpha, absW);
    tmpY = residual;
    estimatedW(r,:) = W_r;     
end
t = toc;
fprintf('Elapsed time is %.4f sec\n',t)
%clear invertX;
% compute errors
estimatedWVec = zeros(1,prod(p(1:M))); 
for r = 1:cvR
    estimatedWVec = estimatedWVec + estimatedW(r,:);
end
predY = zeros(N-splitPoint, 1);
vecX = tenmat(testX, 1);
vecX = vecX.data;
for i = 1:(N-splitPoint)
    predY(i) = vecX(i,:) * estimatedWVec';
end
testY = tensor(testY.data, [N-splitPoint 1]);
testY = testY.data;
[rocX, rocY, rocT, AUC] = perfcurve(double(testY), sigmoid(predY), 1);
fprintf('The AUC value of SURF is %f\n', AUC);
% save the ROC curve
SURF_Roc.X = rocX;
SURF_Roc.Y = rocY;
SURF_Roc.T = rocT;
SURF_Roc.AUC = AUC;
save('SURF_Roc.mat', 'SURF_Roc')

%% Split dataset only for LR
% split into training and testing set
splitPoint = floor(N*0.5);
indices = randperm(N);
trainIndices = indices(1:splitPoint);
testIndices = indices(splitPoint+1:end);
trainY = Y(trainIndices,:);
trainX = X(trainIndices, :,:,:);
%trainInvertX = invertX(:,:,:, trainIndices);
testY = Y(testIndices,:);
testX = X(testIndices, :,:,:);
%testInvertX = invertX(:,:,:, testIndices);

%% Lasso
% vectorize X
clear invertX;
% parameter settings
addpath('GLMNET/');
disp('===== Lasso =====')
disp('Start time')
LassoOpt.alpha = 1;
LassoOpt.nlambda = 1000;
LassoOpt = glmnetSet(LassoOpt);
% time cost
% convet X to vector for Lasso and ENet
vecX = tenmat(trainX, 1);
vecX = vecX.data;
tic
fit = glmnet(vecX, double(trainY), [], LassoOpt);
t = toc;
fprintf('Elapsed time is %.4f sec\n',t)
% find out the best lambda and errors correspondingly
vecX = tenmat(testX, 1);
vecX = vecX.data;
allPredY = glmnetPredict(fit, vecX, [], 'response');
lambdaNum = size(fit.lambda);
lambdaNum = lambdaNum(1);
response_errors = zeros(1, lambdaNum);
for i = 1:lambdaNum
    predY = allPredY(1:end,i);
    response_errors(i) = norm(tensor(double(testY)-predY));
    if i == 1
        minError = response_errors(1);
        minIndex = 1;
    else
        if response_errors(i) < minError
            minError = response_errors(i);
            minIndex = i;
        end
    end
end
predY = allPredY(1:end,minIndex);
predY = reshape(double(predY), [N-splitPoint 1]);
[rocX, rocY, rocT, AUC] = perfcurve(double(testY), sigmoid(predY), 1);
fprintf('The AUC value of Lasso is %f\n', AUC);
% save the ROC curve
Lasso_ROC.X = rocX;
Lasso_ROC.Y = rocY;
Lasso_ROC.T = rocT;
Lasso_ROC.AUC = AUC;
save('Lasso_ROC7.mat', 'Lasso_ROC')
disp('Finish time')

%% ENet
% parameter settings
addpath('GLMNET/');
disp('===== Elasticnet =====')
disp('Start time')
LassoOpt.alpha = 0.5;
LassoOpt.nlambda = 1000;
LassoOpt = glmnetSet(LassoOpt);
% time cost
% convet X to vector
vecX = tenmat(trainX, 1);
vecX = vecX.data;
tic
fit = glmnet(vecX, double(trainY), [], LassoOpt);
t = toc;
fprintf('Elapsed time is %.4f sec\n',t)
% find out the best lambda and errors correspondingly
vecX = tenmat(testX, 1);
vecX = vecX.data;
allPredY = glmnetPredict(fit, vecX, [], 'response');
lambdaNum = size(fit.lambda);
lambdaNum = lambdaNum(1);
response_errors = zeros(1, lambdaNum);
for i = 1:lambdaNum
    predY = allPredY(1:end,i);
    response_errors(i) = norm(tensor(double(testY)-predY));
    if i == 1
        minError = response_errors(1);
        minIndex = 1;
    else
        if response_errors(i) < minError
            minError = response_errors(i);
            minIndex = i;
        end
    end
end
predY = allPredY(1:end,minIndex);
predY = reshape(double(predY), [N-splitPoint 1]);
[rocX, rocY, rocT, AUC] = perfcurve(double(testY), sigmoid(predY), 1);
fprintf('The AUC value of Lasso is %f\n', AUC);
% save the ROC curve
ENet.X = rocX;
ENet.Y = rocY;
ENet.T = rocT;
ENet.AUC = AUC;
save('ENet_ROC7.mat', 'ENet')
disp('Finish time')

