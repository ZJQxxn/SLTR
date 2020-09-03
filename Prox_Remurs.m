%{
Description:
    Sparse Higher-Order Tensor Regression Models with Automatic Rank Explored 

Reference:


Author:
    Jiaqi Zhang <zjqseu@gmail.com>

History:
    
%}
function [estimatedW, errSeq] = Prox_Remurs(X, Y, tau, lambda, epsilon, rho, maxIter, minDiff)
    %% Initializations
    %rho = 1; % TODO: set to 1 at present
    p = size(X);
    prodP = prod(p(1:end-1));
    N = p(end);
    dims = ndims(X);
    M = dims - 1;
    % vectorizing the 3D samples 
    % TODO: take matrix X as an input argument
    vecX = Unfold(X, p, dims);
    % Original approximation: (X^TX + epsilon I)^{-1} * X^Ty
    startApprox = ( (vecX' * vecX + epsilon * eye(prodP)) \ eye(prodP) ) * vecX' * Y;
    startApprox = reshape(startApprox, p(1:M)); % reshape to a tensor
    
    %% Main Procedure
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
           %% Terminatio  check
            errSeq(t) = norm(tensor(lastW-W_m)) / prod(p(1:M));
            if  errSeq(t) <= minDiff
                %fprintf('Algorithm terminates at the %d-th iteration\n', t)
                break
            end
            lastW = W_m;
        end
        if m == 1
            estimatedW = Fold(W_m, p(1:M), m);
        else
            estimatedW = estimatedW +  Fold(W_m, p(1:M), m);
        end
    end
    estimatedW = estimatedW / M;
    
    %predY = zeros(N, 1);
    %for i = 1:N
    %    predY(i) = vecX(i,:) * estimatedW;
    %end
    %mse = norm(tensor(predY-Y))/N;
    %estimatedW = reshape(estimatedW, p(1:M));

    
    
    