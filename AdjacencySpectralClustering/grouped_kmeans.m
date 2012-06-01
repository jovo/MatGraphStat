function [ cluster_idx, cluster_centroid ] = grouped_kmeans( feat, kCluster, start_idx, varargin )
%grouped_kmeans k-means on feat where all vectors corresponding to the
%same node are kept in the same group
%   feat is a nNodes*d*nGraphs set of feature vectors ie one row vector in
%   R^d for each node and graph. We do k-means where we cluster the nodes
%   keeping all feature vectors corresponding to the same node together.
%   cluster_idx is a nNodes*1 column vector with the cluster for each node.
%   cluster_centroid is kClusters*d, ie a row vector for each cluster.

nNodes = size(feat,1);
d = size(feat,2);
nGraphs = size(feat,3);
maxIt = 100;
cluster_idx = start_idx;
cluster_centroid = zeros(kCluster,d);
old_centroid = ones(kCluster,d);

if ~isempty(varargin)
    distance = varargin{1};
else
    distance = @(v,w) norm(v-w);
end

it = 1;
while it<maxIt && norm(old_centroid-cluster_centroid,inf)>1e-5
    it = it+1;
    old_centroid = cluster_centroid;
    
    for k=1:kCluster
        % find mean of all vectors with label k, there will be some
        % multiple of nGraphs of these.
        cluster_centroid(k,:) = mean(mean(feat(cluster_idx==k,:,:),1),3);
    end
    
    for n=1:nNodes
        % assign each node to the cluster with the minimum mean distance
        
        % compute all interpoint distances between graph features at
        % current node and cluster centroids, kCluster x nGraphs, take mean
        distToCentroid = mean(arrayfun(@(v,w) ...
            distance(feat(n,:,v),cluster_centroid(w,:)), ...
            repmat(1:nGraphs,kCluster,1),repmat((1:kCluster)',1,nGraphs)),2);
        [~,cluster_idx(n)] = min(distToCentroid);
    end
end

end

