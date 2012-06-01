function [ cluster_idx, cluster_centroid ] = ...
    cluster_features( feat, kCluster, varargin)
%cluster_features Uses k-means cluster features into kCluster groups 
%   Detailed explanation goes here
n = size(feat,3);
if n==1
    [ cluster_centroid, cluster_idx ] = ...
        cluster_features_one( feat, kCluster, varargin)
    return;
end

d = size(feat,2);
cluster_centroid = zeros(kCluster,d,n);
cluster_idx =  zeros(kCluster,1,n);

for k=1:n
    [cluster_centroid(:,:,k),cluster_idx(:,1,k)] = ...
        cluster_features_one( feat(:,:,k), kCluster, varargin);
end
end

function [ cluster_centroid, cluster_idx ] = ...
    cluster_features_one( feat, kCluster, varargin)

args = varargin{1};
%default to euclidian distance
distance = find(strcmp('distance',args));
if isempty(distance)
    distance = 'euclidean';
else
    distance = args{distance+1};
end

d = size(feat,2);
cluster_centroid = zeros(kCluster,d);
T = clusterdata(feat, 'maxClust', kCluster, 'distance', distance);

for j=1:kCluster
    cluster_centroid(j,:) = mean(feat(T==j,:),1);
end

[idx, cluster_centroid] = kmeans(feat,kCluster, 'start',cluster_centroid, ...
                                    'distance',distance);
                                
cluster_idx = arrayfun(@(v) sum(idx==v),1:kCluster);


end

