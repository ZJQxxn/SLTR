%% Experiment Setups
tauList = [10^-2, 5*10^-2, 10^-1, 5*10^-1, 10^0, 5*10^0, 10^1, 5*10^1];
lambdaList = [10^-4, 5*10^-4, 10^-3, 5*10^-3, 10^-2, 5*10^-2];
for i=1:length(tauList)
    for j=1:length(lambdaList)
        pars{i,j}=[tauList(i), lambdaList(j)];
    end
end
warning off;

for index = 1:(length(tauList)*length(lambdaList))
par=pars{index};
fprintf('======= tau: %f; lambda: %f =======', par(1), par(2))
%% Generate simulated datasets
%clear X W Y Xvec Wvec invertX estimatedW fit;

addpath('tensor_toolbox/');
% parameters for generating datasets¡£
% options.p = [20 20 20];
options.p = [30 30 5];
options.N = 360;
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

%% Experiment settings
repeat =10;

%% Prox_Remurs
% parameter settings
disp('===== Prox_Remurs =====')
tau = par(1);
lambda = par(2);
epsilon = 1;
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
    [estimatedW, errSeq] = Prox_Remurs(double(invertX), matrixY, tau, lambda, epsilon, rho, maxIter, minDiff);
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

end