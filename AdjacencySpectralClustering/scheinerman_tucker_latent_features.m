function X = scheinerman_tucker_latent_features(A,d,epsilon)
n = length(A);
D = zeros(n);
Dold = ones(n);
X = ones(n,d);
Xold = zeros(n,d);

if nargin < 3
    epsilon = 10^-8;
end
k=1;
while norm(D-Dold,'fro')>epsilon
    Xold = X;
    Dold = D;
    [U,Lambda] = eigs(A+D,d,'la');
    Lambda = max(zeros(d),Lambda);
    X =  U*Lambda.^(.5);
    D = diag(diag(X*X'));
    k=k+1;
end
end
