addpath('tensor_toolbox/');
% parameters for generating datasets¡£
% options.p = [20 20 20];
options.p = [30 30 5];
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
disp(options)

%% Experiment settings
repeat =1;

%% Prox_Remurs
%{
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


for it = 1:repeat
    tic
    [estimatedW, errSeq] = Prox_Remurs(double(invertX), matrixY, 1e-4, 1e-4, 0.5, rho, maxIter, minDiff);
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
%}

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

for it = 1:repeat
    tic
    [estimatedW, errList] = Remurs(double(invertX), matrixY, 1e-1, 1e-1, epsilon, iter);
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
