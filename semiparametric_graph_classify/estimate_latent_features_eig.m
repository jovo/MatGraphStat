function [ lat ] = estimate_latent_features_eig( A, d)
%estimate_latent_features_eig Returns estimated (undirected) latent features
%   A is an (or many) nxn (or nxnxk) adjacency matrix). d optionally 
%   indicate the dimensions of the in and out feature vectors.
%   The default is n for each. The feature vectors are estimated using the
%   eig method. lat is d x n x k. 

sz =  size(A);

if nargin == 1
    d = sz(1);
end

% each Laplacian matrix L = V*D*V';
[V D] = graph_eig(graph_laplacian(A));
D = double(D>0).*D;

if length(sz) == 2
    lat =  V(:,1:d)*(D(1:d,1:d).^.5);    
    return;
end

lat = zeros(sz(1),d,sz(3));
for k=1:sz(3) 
    lat(:,:,k) =  V(:,1:d,k)*(D(1:d,1:d,k).^.5);
    if k~=1
     [U,~,R] = svd(lat(:,:,k)'*lat(:,:,1));
     lat(:,:,k) = lat(:,:,k)*R*U';
    end
end    
end

