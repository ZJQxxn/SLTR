function [] = each_iter_exp()
    %% Generate datasets
    addpath('tensor_toolbox/');
    warning off;
    % parameters for generating datasets¡£
    options.p = [30 30 5];
    options.N = 360;
    options.R = 1;
    options.sparsity = 0.8; 
    options.noise_coeff = 0.1;
    M = length(options.p);
    [X, W, Y, Xvec, Wvec, invertX] = sparseGenerate(options);
    disp(options)
    
   %% Prox_Remurs
    % parameter settings
    disp('===== Prox_Remurs =====')
    rho = 0.8; % learning rate
    minDiff=1e-4;
    maxIter=1000;
    % time cost
    totalTime = 0;
    totalMSE = 0;
    totalEE = 0;
    matrixY=reshape(Y.data,[options.N 1]);
    % 5-fold cv
    [cvTau, cvLambda, cvEpsilon] = cv_Prox_Remurs(double(invertX), matrixY, options.p,...
        [0, 10^-2, 5*10^-2, 10^-1, 5*10^-1, 10^0, 5*10^0, 10^1, 5*10^1],...
        [0, 10^-4, 5*10^-4, 10^-3, 5*10^-3, 10^-2, 5*10^-2],...
        [0, 0.1, 0.2, 0.3, 0.4, 0.5],...
        rho, 5, maxIter, minDiff);

    [err_mat, estimatedW, ~] = iter_Prox_Remurs(double(invertX), matrixY, cvTau, cvLambda, cvEpsilon, rho, maxIter, minDiff);
    
    predY = ttt(tensor(X), tensor(estimatedW), 2:M+1, 1:M);
    error = reshape(W.data, options.p)-estimatedW;
    fprintf('Response  Error is %.4f\n',(norm(tensor(predY.data, [options.N, 1]) - tensor(matrixY)) / options.N))
    fprintf('Estimation Error is %.4f\n',(norm(tensor(error)) /norm(tensor(W))))
    save('Prox_Remur_iter.mat', 'err_mat');
    %% Remurs
    clear err_mat;
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

    [cvAlpha, cvBeta] = cv_Remurs(double(invertX), matrixY, options.p,...
        [10^-3, 5*10^-3, 10^-2, 5*10^-2, 10^-1, 5*10^-1, 10^0, 5*10^0, 10^1, 5*10^1, 10^2, 5*10^2],...
        [10^-3, 5*10^-3, 10^-2, 5*10^-2, 10^-1, 5*10^-1, 10^0, 5*10^0, 10^1, 5*10^1, 10^2, 5*10^2],...
        5, iter, epsilon);
    [err_mat, estimatedW, ~] = iter_Remurs(X, double(invertX), matrixY, cvAlpha, cvBeta, epsilon, iter);
    predY = ttt(tensor(X), tensor(estimatedW), 2:M+1, 1:M);
    fprintf('Prediction Error: %f\n',norm(tensor(predY.data, [options.N, 1]) - tensor(matrixY)) / options.N)
    error = reshape(W.data, options.p)-estimatedW;
    fprintf('Estimation error: %f\n',norm(tensor(error)) /norm(tensor(W)))
    save('Remurs_iter.mat','err_mat');
    
    
function [err_mat, estimatedW, errSeq] = iter_Prox_Remurs(X, Y, tau, lambda, epsilon, rho, maxIter, minDiff)
    %% Initializations
    p = size(X);
    prodP = prod(p(1:end-1));
    N = p(end);
    dims = ndims(X);
    M = dims - 1;
    % vectorizing the 3D samples 
    vecX = Unfold(X, p, dims);
    % Original approximation: (X^TX + epsilon I)^{-1} * X^Ty
    startApprox = ( (vecX' * vecX + epsilon * eye(prodP)) \ eye(prodP) ) * vecX' * Y;
    startApprox = reshape(startApprox, p(1:M)); % reshape to a tensor
    
    %% Main Procedure
    err_mat=zeros([M maxIter 2]);
    for m=1:M
        z_m = Unfold(startApprox, p(1:M), m);
        for i = 1:4
            W_t{i} = z_m;
        end
        W_m = z_m;
        lastW = W_m;
        for t=1:maxIter
            a_t{1} = l1_prox(W_t{1}, 4*lambda);
            a_t{2} = inf_set_prox(W_t{2}, z_m, 4*lambda);
            a_t{3} = nuclear_prox(W_t{3}, 4*tau);
            a_t{4} = spec_set_prox(W_t{4}, z_m, 4*tau);
            a = (a_t{1} + a_t{2} + a_t{3} + a_t{4}) / 4; % a^t
            for i = 1:4
                W_t{i} = W_t{i} + rho*(2*a-W_m-a_t{i});
            end
            W_m = W_m + rho*(a-W_m);
         
            errSeq(t) = norm(tensor(lastW-W_m)) / norm(tensor(lastW));
            l1_norm = 0;
            for index=1:numel(W_m)
                l1_norm = l1_norm + abs(W_m(index));
            end
            nuclear_norm = norm(svd(W_m), 1);
            objFunc = l1_norm + nuclear_norm;
            err_mat(m,t,:)=[errSeq(t), objFunc];
            lastW = W_m;
        end
        if m == 1
            estimatedW = Fold(W_m, p(1:M), m);
        else
            estimatedW = estimatedW +  Fold(W_m, p(1:M), m);
        end
    end
    estimatedW = estimatedW / M;
    

function [err_mat, tW, errList, time] = iter_Remurs(X, tX, y, alpha, beta, epsilon, max_iter)
%% Set defalt parameters
if nargin < 4
    fprintf('Not enough parameters for Remurs! Terminate now.\n')
    return;
end

if nargin < 6
    max_iter = 1000;
end

if nargin < 5
    epsilon = 1e-4;
end

%% Initialize
lambda   = 1;
rho      = 1/lambda;
N        = ndims(tX) - 1; % Number of modes.
size_tX  = size(tX);
dim      = size_tX(1:N);
X        = Unfold(tX, size(tX), N+1); % Data in vector form. Matrix: M*d.
Xty      = X'*y;
numFea   = size(X,2);
numSam   = size(X,1);
tW        = zeros([dim,1]); % Tensor W
tU        = zeros([dim,1]); % Tensor U
tA        = zeros([dim,1]); % Tensor A
for n = 1:N
    tV{n} = zeros([dim,1]); % Tensor V_n
    tB{n} = zeros([dim,1]); % Tensor B_n
end

[L U] = factor(X, rho);

err_mat=[]; % [Jiaqi Zhang] revised
%% Iterate: Main algorithm
tic;
for k = 1:max_iter
    % Update tU: quadratic proximal operator
    q = Xty + rho*(tW(:) - tA(:));
    if numSam >= numFea
        u = U \ (L \ q);
    else
        u = lambda*(q - lambda*(X'*(U \ ( L \ (X*q) ))));
    end    
    
    tU = reshape(u, [dim,1]);
    
    % Update tV_n: trace-norm proximal operator
    for n = 1:N
        tV{n} = Fold( prox_nuclear( Unfold(tW-tB{n}, dim, n), alpha/rho/N ), dim, n );
    end
    
    % Update tW: l1-norm proximal operator
    last_tW = tW;
    tSum_uv = tU;
    tSum_ab = tA;
    for n = 1:N
        tSum_uv = tSum_uv + tV{n};
        tSum_ab = tSum_ab + tB{n};
    end
    tW =  prox_l1( (tSum_uv+tSum_ab)/(N+1), beta/(N+1)/rho );
    
    % Update tA
    tA = tA + tU - tW;
    
    % Update tB_n
    for n = 1:N
        tB{n} = tB{n} + tV{n} - tW;
    end
    
    % Check termination
    errList(k) = norm(tW(:)-last_tW(:)) / norm(tW(:));
    objFunc = 0;
    predY = ttt(tensor(tX), tensor(tW), 1:N, 1:N);
    pred_err = norm(tensor(predY.data, [size_tX(N+1), 1]) - tensor(y)) / 2;
    l1_norm = 0;
    for index =1:numel(tW)
        l1_norm = l1_norm + abs(tW(index));
    end
    nuclear_norm = 0;
    for n = 1:N
        tempW = Unfold(tW, dim, n);
        nuclear_norm = norm(svd(tempW),1);
    end
    nuclear_norm = nuclear_norm / N;
    objFunc = pred_err + beta*l1_norm + alpha*nuclear_norm;
    err_mat=[err_mat;[errList(k) objFunc]];
end
time = toc;