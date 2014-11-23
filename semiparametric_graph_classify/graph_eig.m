function [V D] = graph_eig(L)
%graph_eig Computes the eigenvalue Decomp of the Laplacian of an adjacancy 
% matrix or matrices
%   Returns the V,D for matrix or matrices for each Laplacian matrix
%   in L. Ie L can be either a 2d square matrix or a 3d array of square
%   matrices. Each laplacian willhave the property L = V*D*V';
d =  size(L);
rev = d(1):-1:1;
if numel(d) == 2
    [V D] = eig(L);
    V = V(:,rev);
    D = D(rev,rev);
    return;
end
V = zeros(d);
D = V;
for k=1:d(3)
    [V(:,:,k) D(:,:,k)] = eig(L(:,:,k));
    V(:,:,k) = V(:,rev,k);
    D(:,:,k) = D(rev,rev,k);
end

end

