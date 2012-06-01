function [ logP ] = log_prob_graph_given_lat_feat( A, feat )
%log_prob_graph_given_lat_feat Compute probability of seeing the Adjacency
%matrix A given the latent features feat.
%   A is nxn binary adjacency matrix. Feat is nxd matrix of n row vectors
%   in R^d. P(A(u,v) = 1) = feat(u,:)*feat(v,:)'
logP = 0;
n = size(A,1);
for i=1:n-1
    for j=i+1:n
        logP = logP+log(A(i,j)*(feat(i,:)*feat(j,:)')...
                   +(1-A(i,j))*(1-feat(i,:)*feat(j,:)'));
        if imag(logP) ~= 0
            error(['Log is Imag after edge ',num2str(i),' to ',num2str(j),',',num2str(feat(i,:)*feat(j,:)')]);
        end
    end
end

end