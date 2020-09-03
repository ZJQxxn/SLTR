%{
Description:
    Cross-validation for Remurs.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

History:
    
%}
function [cvAlpha, cvBeta] = cv_Remurs(X, Y, p, alphaList, betaList,fold, iter, epsilon)
    addpath('tensor_toolbox/')
    addpath('RemursCode/')
    if isempty(iter)
        iter = 1000;
    end
    if isempty(epsilon)
        diff = 1e-3;
    end
    M = length(p);
    % initialize the parameter pairs
    parameterPair = {};
    for i = 1:length(alphaList)
        for j = 1:length(betaList)
                parameterPair{i,j} = [alphaList(i) betaList(j)];
        end
    end
    % cross-validation initialization  
    N = size(X);
    N = N(end);
    cvp = cvpartition(N, 'Kfold', fold);
    for t = 1:length(alphaList)*length(betaList)
        testErr = 0.0;
        for f = 1:cvp.NumTestSets
            pars = parameterPair{t};
            trains = cvp.training(f);
            tests =  cvp.test(f);
            trainIndex = [];
            testIndex = [];
            for i = 1:N
                if trains(i) == 1
                    trainIndex = [trainIndex i];
                end
                if tests(i) == 1
                    testIndex = [testIndex i];
                end
            end
            % 3-D variates 
            Xtrain = X(:,:,:,trainIndex);
            Ytrain = Y(trainIndex,:);
            Xtest = X(:,:,:,testIndex);
            Ytest = Y(testIndex,:);
            [estimatedW, ~] = Remurs(double(Xtrain), double(Ytrain), pars(1), pars(2), epsilon, iter);
            % compute MSE
            predY = ttt(tensor(Xtest), tensor(estimatedW), 1:M, 1:M);
            testErr = testErr + norm(tensor(predY.data, [cvp.TestSize(f) 1]) - tensor(Ytest)) / cvp.TestSize(f);
        end
        testErr = testErr / fold;
        % update the best setting of parameters
        if t == 1
            minErr = testErr;
            bestPair = parameterPair{t};
        else
            if testErr < minErr
                minErr = testErr;
                bestPair = parameterPair{t};
            end
        end
    end
    cvAlpha = bestPair(1);
    cvBeta = bestPair(2);
    fprintf('cvAlpha : %f; cvBeta : %f\n', cvAlpha, cvBeta)
   
        