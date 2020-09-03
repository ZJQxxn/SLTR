function [u] = spec_set_prox(x, z, tau)
    [U,S,V] = svd(x-z,'econ');
    eigVal = diag(S);
    if max(max(S)) <= tau
        u = z;
    else
        for i = 1:length(eigVal)
            if eigVal(i) > tau
                eigVal(i) = tau;
            end
        end
        u = U*diag(eigVal)*V' + z;
    end