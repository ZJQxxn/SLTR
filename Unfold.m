%{
Description:
    Unfold a tensor into a vector.

Reference:


Author:
    Jiaqi Zhang <zjqseu@gmail.com>

History:
    
%}
function [vecX] = Unfold( X, dim, i )
vecX = reshape(shiftdim(X,i-1), dim(i), []);
