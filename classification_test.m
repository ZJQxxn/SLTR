warning off;

%% Generate simulated datasets
%clear X W Y Xvec Wvec invertX estimatedW fit;
addpath('tensor_toolbox/');
addpath('fMRI_data/');
% parameters for generating datasets¡£
% options.p = [20 20 20];
options.p = [10 10 5];
%options.N = 100;
options.N = 50;
options.R = 1;
options.sparsity = 0.8; 
options.noise_coeff = 0.1;
M = length(options.p);
%{
    X -- a tensor with shape N x p1 x p2 x...x pM 
    W -- a tensor with shape p1 x p2 x...x pM 
    Y -- a trensor with shape N
    Xvec -- a matrix with shape N x (p1 * p2 *...* pM)
    Wvec -- a vector with shape (p1 * p2 *..* pM) x 1 
    invertX -- a tensor with shape p1 x p2 x...x pM x N
%}
%[X, W, Y, Xvec, Wvec, invertX] = generateData(options);
[X, W, Y, Xvec, Wvec, invertX] = sparseGenerate(options);
for i = 1:length(Y.data)
    temp = rand();
    if temp > 0.5
        Y(i) = 1;
    else
        Y(i) = 0;
    end   
end
disp(options)

%% Prox_Remurs
% parameter settings
disp('===== Prox_Remurs =====')
rho = 0.8; % learning rate
minDiff=1e-4;
maxIter=1000;
matrixY=reshape(Y.data,[options.N 1]);
cvTau = 1e-3;
cvLambda = 1e-3;
cvEpsilon = 0.5;
% main step
tic
[estimatedW, errSeq] = Prox_Remurs(double(invertX), matrixY, cvTau, cvLambda, cvEpsilon, rho, maxIter, minDiff);
t = toc;
fprintf('Elapsed time is %.4f sec\n',t)
% ROC and AUC
predY = ttt(tensor(X), tensor(estimatedW), 2:M+1, 1:M); 
predY = reshape(double(predY),[options.N 1]);
[rocX, rocY, rocT, AUC] = perfcurve(double(Y), sigmoid(predY), 1);
fprintf('The AUC value of Prox_Remurs is %f\n', AUC);
% save the ROC curve
Prox_ROC.X = rocX;
Prox_ROC.Y = rocY;
Prox_ROC.T = rocT;
Prox_ROC.AUC = AUC;
%save('Prox_ROC.mat', 'Prox_ROC')

%% Remurs
% parameter settings
addpath('RemursCode/Code/')
disp('===== Remurs =====')
setting = expSet;
epsilon=1e-4;
iter=1000;
matrixY=reshape(Y.data,[options.N 1]);
cvAlpha = 1e-3;
cvBeta = 1e-3;
% main step
tic
[estimatedW, errList] = Remurs(double(invertX), matrixY, cvAlpha, cvBeta, epsilon, iter);
t = toc;
fprintf('Elapsed time is %.4f sec\n',t)
predY = ttt(tensor(X), tensor(estimatedW), 2:M+1, 1:M);
predY = reshape(double(predY),[options.N 1]);
[rocX, rocY, rocT, AUC] = perfcurve(double(Y), sigmoid(predY), 1);
fprintf('The AUC value of Remurs is %f\n', AUC);
% save the ROC curve
Remur_ROC.X = rocX;
Remur_ROC.Y = rocY;
Remur_ROC.T = rocT;
Remur_ROC.AUC = AUC;
%save('Remur_ROC.mat', 'Remur_ROC')

%% Lasso
% vectorize X
load('P3_X_mat.mat')
% parameter settings
testRatio = 0.2;
testIndex = floor(testRatio * options.N);
addpath('GLMNET/');
disp('===== Elasticnet =====')
LassoOpt.alpha = 1;
LassoOpt.nlambda = 1000;
LassoOpt = glmnetSet(LassoOpt);
% time cost
tic
fit = glmnet(Xvec(1:end-testIndex,:), double(Y(1:end-testIndex)), [], LassoOpt);
t = toc;
fprintf('Elapsed time is %.4f sec\n',t)
% find out the best lambda and errors correspondingly
allPredY = glmnetPredict(fit, Xvec(end-testIndex+1:end,:), [], 'response');
lambdaNum = size(fit.lambda);
lambdaNum = lambdaNum(1);
response_errors = zeros(1, lambdaNum);
for i = 1:lambdaNum
    predY = allPredY(1:end,i);
    response_errors(i) = norm(tensor(double(Y(end-testIndex+1:end))-predY)) / testIndex;
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
predY = reshape(double(predY), [testIndex 1]);
[rocX, rocY, rocT, AUC] = perfcurve(double(Y(end-testIndex+1:end)), sigmoid(predY), 1);
fprintf('The AUC value of Lasso is %f\n', AUC);
% save the ROC curve
Lasso_ROC.X = rocX;
Lasso_ROC.Y = rocY;
Lasso_ROC.T = rocT;
Lasso_ROC.AUC = AUC;
%save('Lasso_ROC.mat', 'Lasso_ROC')

%% ENet
% parameter settings
testRatio = 0.2;
testIndex = floor(testRatio * options.N);
addpath('GLMNET/');
disp('===== Elasticnet =====')
LassoOpt.alpha = 0.5;
LassoOpt.nlambda = 1000;
LassoOpt = glmnetSet(LassoOpt);
% time cost
tic
fit = glmnet(Xvec(1:end-testIndex,:), double(Y(1:end-testIndex)), [], LassoOpt);
t = toc;
fprintf('Elapsed time is %.4f sec\n',t)
% find out the best lambda and errors correspondingly
allPredY = glmnetPredict(fit, Xvec(end-testIndex+1:end,:), [], 'response');
lambdaNum = size(fit.lambda);
lambdaNum = lambdaNum(1);
response_errors = zeros(1, lambdaNum);
for i = 1:lambdaNum
    predY = allPredY(1:end,i);
    response_errors(i) = norm(tensor(double(Y(end-testIndex+1:end))-predY)) / testIndex;
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
predY = reshape(double(predY), [testIndex 1]);
[rocX, rocY, rocT, AUC] = perfcurve(double(Y(end-testIndex+1:end)), sigmoid(predY), 1);
fprintf('The AUC value of ENEt is %f\n', AUC);
% save the ROC curve
Lasso.X = rocX;
Lasso.Y = rocY;
Lasso.T = rocT;
Lasso.AUC = AUC;
%save('ENet_ROC.mat', 'ENet')

%% SURF
% parameter settings
addpath('SURF_code/')
addpath('SURF_code/tensorlab/')
disp('===== SURF =====')
%epsilon = 0.1;
%xi = epsilon^2 / 2; % [Jiaqi Zhang] set to the value recomended in the paper
%alpha = 1;
absW = 1e-3;
[cvAlpha, cvEpsilon, cvR] = cv_SURF(double(invertX), Xvec, double(Y), options.p,...
    [1e-1],...
    [1e-2],...
    [1],...
    5, absW);
% time cost
totalTime = 0;
tmpY = Y;
% main procedure
estimatedW = zeros(cvR,prod(options.p));
tic
for r =1:cvR
    [W_r, residual] = MyTrain(double(invertX), Xvec, double(tmpY), cvEpsilon, cvEpsilon^2/2, cvAlpha, absW);
    tmpY = residual;
    estimatedW(r,:) = W_r;     
end
t = toc;
fprintf('Elapsed time is %.4f sec\n',t)
clear invertX;
% compute errors
estimatedWVec = zeros(1,prod(options.p)); 
for r = 1:cvR
    estimatedWVec = estimatedWVec + estimatedW(r,:);
end
error = Wvec'-estimatedWVec;
predY = zeros(options.N, 1);
for i = 1:options.N
    predY(i) = Xvec(i,:) * estimatedWVec';
end
Y = tensor(Y.data, [options.N 1]);
Y = Y.data;
[rocX, rocY, rocT, AUC] = perfcurve(double(Y), sigmoid(predY), 1);
fprintf('The AUC value of SURF is %f\n', AUC);
% save the ROC curve
SURF.X = rocX;
SURF.Y = rocY;
SURF.T = rocT;
SURF.AUC = AUC;
%save('Prox_ROC.mat', 'Prox_ROC')