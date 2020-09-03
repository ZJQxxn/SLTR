function x = l1_prox(v, lambda)
% PROX_L1    The proximal operator of the l1 norm.
%
%   prox_l1(v,lambda) is the proximal operator of the l1 norm
%   with parameter lambda.
    x = v;
    for i=1:numel(v)
        if abs(v(i)) <= lambda
            x(i) = 0;
        end
        if v(i) > lambda
            x(i) = v(i)-lambda;
        end
        if v(i) < -lambda
            x(i) = v(i)+lambda;
        end
    end
end
