%{
Description:
    Sparse Higher-Order Tensor Regression Models with Automatic Rank Explored 

Reference:


Author:
    Jiaqi Zhang <zjqseu@gmail.com>

History:
    
%}
function x = nuclear_prox(v, tau)
%   Evaluates the proximal operator of the nuclear norm at v
%   (i.e., the singular value thresholding operator).

    [U,S,V] = svd(v,'econ');
    %[U,S,V] = MySVD(v);
    x = U*diag(l1_prox(diag(S), tau))*V';
end