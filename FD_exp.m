%% Experiment Setups
pList  = [...
    [10 10 5 5];...
    ];
N = [200]; %0.08

warning off;
addpath('tensor_toolbox/');
addpath('4D_test/')

for i = 1:length(pList)
    
%% Generate simulated datasets
%clear X W Y Xvec Wvec invertX estimatedW fit;
% parameters for generating datasets¡£
% options.p = [20 20 20];
options.p = pList(i,:);
%options.N = 100;
options.N = N(i);
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
[X, W, Y, Xvec, Wvec, invertX] = sparse4DGenerate(options);
disp(options)

%% Experiment settings
repeat =10;

%% Remurs

% parameter settings
addpath('RemursCode/Code/')
disp('===== Remurs =====')
setting = expSet;
epsilon=1e-4;
iter=1000;
% time cost
totalTime = 0;
totalMSE = 0;
totalEE = 0;

matrixY=reshape(Y.data,[options.N 1]);

[cvAlpha, cvBeta] = cv_4D_Remurs(double(invertX), matrixY, options.p,...
    [10^-3, 5*10^-3, 10^-2, 5*10^-2, 10^-1, 5*10^-1, 10^0, 5*10^0, 10^1, 5*10^1, 10^2, 5*10^2],...
    [10^-3, 5*10^-3, 10^-2, 5*10^-2, 10^-1, 5*10^-1, 10^0, 5*10^0, 10^1, 5*10^1, 10^2, 5*10^2],...
    5, iter, epsilon);

for it = 1:repeat
    tic
    [estimatedW, errList] = Remurs(double(invertX), matrixY, cvAlpha, cvBeta, epsilon, iter);
    t = toc;
    totalTime = totalTime + t;
    predY = ttt(tensor(X), tensor(estimatedW), 2:M+1, 1:M);
    totalMSE = totalMSE + (norm(tensor(predY.data, [options.N, 1]) - tensor(matrixY)) / options.N);
    error = reshape(W.data, options.p)-estimatedW;
    totalEE = totalEE + (norm(tensor(error)) /norm(tensor(W)));
    %totalEE = totalEE + (norm(tensor(error)) / prod(options.p)); 
end
fprintf('Elapsed time is %.4f sec\n',totalTime / repeat)
fprintf('Response  Error is %.4f\n',totalMSE / repeat)
fprintf('Estimation Error is %.4f\n',totalEE / repeat)

%% Prox_Remurs
% parameter settings
disp('===== Prox_Remurs =====')
%tau = 0.5;
%lambda = 5e-3;
%epsilon = 0.8;
rho = 0.8; % learning rate
minDiff=1e-4;
maxIter=1000;
% time cost
totalTime = 0;
totalMSE = 0;
totalEE = 0;
matrixY=reshape(Y.data,[options.N 1]);

[cvTau, cvLambda, cvEpsilon] = cv_4D_Prox_Remurs(double(invertX), matrixY, options.p,...
    [0, 10^-2, 5*10^-2, 10^-1, 5*10^-1, 10^0, 5*10^0, 10^1, 5*10^1],...
    [0, 10^-4, 5*10^-4, 10^-3, 5*10^-3, 10^-2, 5*10^-2],...
    [0, 0.1, 0.2, 0.3, 0.4, 0.5],...
    rho, 5, maxIter, minDiff);

for it = 1:repeat
    tic
    [estimatedW, errSeq] = Prox_Remurs(double(invertX), matrixY, cvTau, cvLambda, cvEpsilon, rho, maxIter, minDiff);
    t = toc;
    totalTime = totalTime + t;
    predY = ttt(tensor(X), tensor(estimatedW), 2:M+1, 1:M);
    totalMSE = totalMSE + (norm(tensor(predY.data, [options.N, 1]) - tensor(matrixY)) / options.N);
    error = reshape(W.data, options.p)-estimatedW;
    totalEE = totalEE + (norm(tensor(error)) /norm(tensor(W)));
    %totalEE = totalEE + (norm(tensor(error)) / prod(options.p)); 
end
fprintf('Elapsed time is %.4f sec\n',totalTime / repeat)
fprintf('Response  Error is %.4f\n',totalMSE / repeat)
fprintf('Estimation Error is %.4f\n',totalEE / repeat)

%% Lasso
% parameter settings
testRatio = 0.2;
testIndex = floor(testRatio * options.N);
addpath('GLMNET/');
disp('===== Lasso =====')
LassoOpt.alpha = 1;
LassoOpt.nlambda = 1000;
%LassoOpt.lambda_min = 0.05;
LassoOpt = glmnetSet(LassoOpt);
totalTime = 0;
% time cost
for it = 1:repeat
    tic
    fit = glmnet(Xvec(1:end-testIndex,:), double(Y(1:end-testIndex)), [], LassoOpt);
    t = toc;
    totalTime = totalTime + t;
end
fprintf('Elapsed time is %.4f sec\n',totalTime / repeat)
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
fprintf('Response Error is %.4f \n', response_errors(1,minIndex))
predW = fit.beta(1:end,minIndex);
fprintf('Estimation error is %.4f \n',norm(tensor(predW-Wvec))/norm(tensor(Wvec)))
%fprintf('Estimation error is %.4f \n',norm(tensor(predW-Wvec))/prod(options.p))

%% Elasticnet (vectorize X)
% parameter settings
testRatio = 0.2;
testIndex = floor(testRatio * options.N);
addpath('GLMNET/');
disp('===== Elasticnet =====')
LassoOpt.alpha = 0.5;
LassoOpt.nlambda = 1000;
%LassoOpt.lambda_min = 0.05;
LassoOpt = glmnetSet(LassoOpt);
totalTime = 0;
% time cost
for it = 1:repeat
    tic
    fit = glmnet(Xvec(1:end-testIndex,:), double(Y(1:end-testIndex)), [], LassoOpt);
    t = toc;
    totalTime = totalTime + t;
end
fprintf('Elapsed time is %.4f sec\n',totalTime / repeat)
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
fprintf('Response Error is %.4f \n', response_errors(minIndex))
predW = fit.beta(1:end,minIndex);
%predW = predW';
fprintf('Estimation error is %.4f \n',norm(tensor(predW-Wvec))/norm(tensor(Wvec)))
%fprintf('Estimation error is %.4f \n',norm(tensor(predW-Wvec))/prod(options.p))

%% SURF
% parameter settings
addpath('SURF_code/')
addpath('SURF_code/tensorlab/')
disp('===== SURF =====')
%epsilon = 0.1;
%xi = epsilon^2 / 2; % [Jiaqi Zhang] set to the value recomended in the paper
%alpha = 1;
absW = 1e-3;
[cvAlpha, cvEpsilon, cvR] = cv_4D_SURF(double(invertX), Xvec, double(Y), options.p,...
    [5e-4 1e-4 5e-3 1e-2 5e-2 1e-1 5e-1 1],...
    [5e-4 1e-4 5e-3 1e-2 5e-2 1e-1 5e-1 1],...
    [1 2],...
    5, absW);
% time cost
totalTime = 0;
tmpY = Y;
for it = 1:repeat
    estimatedW = zeros(cvR,prod(options.p));
    tic
    for r =1:cvR
        [W_r, residual] = MyTrain(double(invertX), Xvec, double(tmpY), cvEpsilon, cvEpsilon^2/2, cvAlpha, absW);
        tmpY = residual;
        estimatedW(r,:) = W_r;     
    end
    t = toc;
    totalTime = totalTime + t; 
end
fprintf('Elapsed time is %.4f sec\n',totalTime / repeat)
% compute errors
estimatedWVec = zeros(1,prod(options.p)); 
for r = 1:cvR
    estimatedWVec = estimatedWVec + estimatedW(r,:);
end
error = Wvec'-estimatedWVec;
predY = zeros(options.N, 1);
%vecX = tenmat(invertX, 4);
vecX = tenmat(X, 1);
vecX = vecX.data;
for i = 1:options.N
    predY(i) = vecX(i,:) * estimatedWVec';
end
Y = tensor(Y.data, [options.N 1]);
Y = Y.data;
fprintf('Response Error is %.6f\n',norm(tensor(predY - Y)) / options.N)
fprintf('Estimation Error is %.6f\n',norm(tensor(error)) / norm(tensor(Wvec)))

end