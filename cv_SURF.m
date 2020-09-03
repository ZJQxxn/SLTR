%{
Description:
    Cross-validation for SURF.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

History:
    
%}
function [cvAlpha, cvEpsilon, cvR] = cv_SURF(X, Xvec, Y, p, alphaList, epsilonList,...
    RList, fold, absW)
    addpath('SURF_code/')
    addpath('tensor_toolbox/')
    if isempty(absW)
        absW = 0.1;
    end
    % initialize the parameter pairs
    parameterPair = {};
    for i = 1:length(alphaList)
        for j = 1:length(epsilonList)
            for k = 1:length(RList)
                parameterPair{i,j,k} = [alphaList(i) epsilonList(j) RList(k) epsilonList(j)^2/2];
            end
        end
    end
    % cross-validation initialization  
    N = size(X);
    N = N(end);
    cvp = cvpartition(N, 'Kfold', fold);
    for t = 1:length(alphaList)*length(epsilonList)*length(RList)
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
            Xvectrain = Xvec(trainIndex,:);
            Ytrain = Y(trainIndex);
            Xtest = X(:,:,:,testIndex);
            Ytest = Y(testIndex);
            % training
            R = pars(3);
            estimatedW = zeros(R,prod(p));
            res = Ytrain;
            for r =1:R
                [W_r, residual] = MyTrain(Xtrain, Xvectrain, double(res), pars(2), pars(4), pars(1), absW);
                res = residual;
                estimatedW(r,:) = W_r;
            end
            % response error
            estimatedWVec = zeros(1,prod(p)); 
            for r = 1:R
                estimatedWVec = estimatedWVec + estimatedW(r,:);
            end
            predY = zeros(cvp.TestSize(f), 1);
            vecX = tenmat(Xtest, 4); % NOTES: fro 3D variates
            vecX = vecX.data;
            for i = 1:cvp.TestSize(f)
                predY(i) = vecX(i,:) * estimatedWVec';
            end
            testErr = norm(tensor(predY - Ytest)) / cvp.TestSize(f);
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
    cvEpsilon = bestPair(2);
    cvR = bestPair(3);
    fprintf('cvAlpha : %f; cvEpsilon : %f; cvR : %d\n', cvAlpha, cvEpsilon, cvR)
   
        