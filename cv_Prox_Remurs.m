%{
Description:
    Cross-validation for EE_Remurs.
    
%}
function [cvTau, cvLambda, cvEpsilon] = cv_Prox_Remurs(X, Y, p, tauList, lambdaList, epsilonList, rho, fold, maxIter, minDiff)
    addpath('tensor_toolbox/')
    %addpath('RemursCode/')
    if isempty(maxIter)
        iter = 1000;
    end
    if isempty(minDiff)
        minDiff = 1e-3;
    end
    M = length(p);
    % initialize the parameter pairs
    parameterPair = {};
    for i = 1:length(tauList)
        for j = 1:length(lambdaList)
            for k = 1:length(epsilonList)
                parameterPair{i,j,k} = [tauList(i) lambdaList(j), epsilonList(k)];
            end
        end
    end
    % cross-validation initialization  
    N = size(X);
    N = N(end);
    cvp = cvpartition(N, 'Kfold', fold);
    for t = 1:length(tauList)*length(lambdaList)*length(epsilonList)
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
            [estimatedW, ~] = Prox_Remurs(double(Xtrain), double(Ytrain), pars(1), pars(2), pars(3), rho, maxIter, minDiff);
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
    cvTau = bestPair(1);
    cvLambda = bestPair(2);
    cvEpsilon = bestPair(3);
    fprintf('cvTau : %f; cvLambda : %f; cvEpsilon : %f\n', cvTau, cvLambda, cvEpsilon)
   
        