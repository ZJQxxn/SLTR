function [u] = inf_set_prox(x, z, lambda)
    p = numel(x); % prod(size(a))
    u = zeros(size(x));
    for i = 1:p
        if abs(x(i)-z(i)) <= lambda
            u(i) = x(i);
        end
        if (x(i)-z(i)) > lambda
            u(i) = z(i) + lambda;
        end
        if (x(i)-z(i)) < -lambda
            u(i) = z(i) - lambda;
        end
    end
    