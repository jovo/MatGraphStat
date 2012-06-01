function L = unnormalized_laplacian(A)
d = size(A);
L=-A;
if numel(d)==2
    for k=1:d(1)
        L(k,k) = sum(A(k,:));
    end
end
if numel(d) == 3
    for g=1:d(3)
        for k=1:d(1)
            L(k,k,g) = sum(A(k,:,g));
        end
    end
end
end

