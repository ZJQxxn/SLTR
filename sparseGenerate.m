function [X, W, Y, Xvec, Wvec, invertX] = sparseGenerate (options)
    addpath('tensor_toolbox/');
    p = options.p;
    R = options.R;
    N = options.N;
    sparsity = options.sparsity;
    noise_coeff = options.noise_coeff;
    M = length(p);
    fprintf('Dims: %d\n', M)
    % Generate true coefficients 
    for r = 1:R
        W_r = tenrand(p);
        %{
        zerosIndex = randperm(prod(p),floor(prod(p)*sparsity)); % randomly generate numbers
        for i=1:length(zerosIndex)
            W_r(zerosIndex(i))=0;
        end
        %}
        if r == 1 
            W = W_r;
        else
            W = W + W_r;
        end
    end
    
    zerosIndex = randperm(prod(p),floor(prod(p)*sparsity)); % randomly generate numbers
    for i=1:length(zerosIndex)
        W(zerosIndex(i))=0;
    end
    
    disp('The size of coefficients is')
    disp(size(W))
    % Generate tensor covariates (centralized)
    X = randn([N prod(p)]);
    for col=1:prod(p)
        X(:,col) = (X(:,col)-mean(X(:,col))) / std(X(:,col));
    end
    X = tensor(reshape(X, [N, p]));
    disp('The size of samples is')
    disp(size(X))
    % invert X for Remurs (3-D variables)
    invertX = tenzeros([p N]);
    for m = 1:M
        invertX(:,:,:,m) = X(m,:,:,:);
    end
    disp('The size of invert samples is')
    disp(size(invertX))
    % Generate responses (centralized)
    Y = ttt(X, W, 2:M+1, 1:M);
    %err = noise_coeff * tenrand([N 1]);
    err = tensor(noise_coeff * normrnd(0,1,[N 1]));
    err = squeeze(err);
    Y = Y + err;
    %Y = (Y.data - mean(Y.data)) / std(Y.data);
    %Y = tensor(Y);
    %Y= squeeze(Y);
    disp('The size of responses is')
    disp(size(Y))
    % Convert ``X'' and ``W'' from tensor-form to vector-form, necessary when using SURF and Lasso
    Xvec = tenmat(X, 1);
    Xvec = Xvec.data;
    Wvec = ttt(tenones([1 2]), W);
    Wvec = squeeze(Wvec);
    Wvec = tenmat(Wvec, 1);
    Wvec = Wvec.data;
    Wvec = Wvec(1,:);
    Wvec = Wvec';
    disp('The size of vector-form X is')
    disp(size(Xvec))
    disp('The size of vector-form W is')
    disp(size(Wvec))