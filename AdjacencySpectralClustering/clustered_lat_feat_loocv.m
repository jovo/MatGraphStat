function [ outStruct ] = clustered_lat_feat_loocv( As, targs, d, kCluster )
%clustered_lat_feat_loocv Classify graphs using bayes plugin based on
%clustered estimated latent features.
%   Estimate latent features using eig method. Cluster the targs==0 and
%   targs==1 groups of features into kCluster clusters giving 2 sets of
%   latent features. Compute log(P(A|Z_i)) where Z_i are the features.

disp(['Clusters:',num2str(kCluster),', LatD:',num2str(d)]);

n = length(targs);

%lat = estimate_latent_features_eig(As, d);

all_ind = struct('ytrn', 1:n, ...
                 'y0trn', find(targs == 0),...  
                 'y1trn', find(targs == 1));
ind = all_ind;
yHat = zeros(size(targs));
logProb0 = zeros(size(targs));
logProb1 = zeros(size(targs));

cN = cell(n,1);
outStruct = struct('lat0',cN,'lat1',cN,'test',cN,...
    'lhat',cN,'yhat',cN,'p0',cN,'p1',cN,'kCluster',cN,'latDim',cN,'targs',cN);
            
for k=1:n
    ind.ytrn = all_ind.ytrn(all_ind.ytrn~=k);
    ind.y0trn = all_ind.y0trn(all_ind.y0trn~=k);
    ind.y1trn = all_ind.y1trn(all_ind.y1trn~=k);
    
    p0 = length(ind.y0trn)/length(ind.ytrn);
    p1 = length(ind.y1trn)/length(ind.ytrn);
    
    lat0 = estimate_latent_features_eig(As(:,:,ind.y0trn),d);
    cluster_idx0 = clusterdata(lat0(:,:,1),kCluster);
    [cluster_idx0, cluster_centroid0] =  ...
        grouped_kmeans(lat0, kCluster, cluster_idx0);

    lat0 = cluster_centroid0(cluster_idx0,:);
    
    lat1 = estimate_latent_features_eig(As(:,:,ind.y1trn),d);
    cluster_idx1 = clusterdata(lat1(:,:,1),kCluster);
    [cluster_idx1, cluster_centroid1] =  ...
        grouped_kmeans(lat1, kCluster, cluster_idx1);    
    
    lat1 = cluster_centroid1(cluster_idx1,:);
    
    % if we get negative probs then lets keep going anyway and see what
    % happens
    try
        logProb0(k) = log_prob_graph_given_lat_feat(As(:,:,k),lat0);
        logProb1(k) = log_prob_graph_given_lat_feat(As(:,:,k),lat1);
        yHat(k) = double(logProb0(k)+log(p0) <logProb1(k)+log(p1));
        outStruct(k).yhat = yHat(k);
        outStruct(k).lhat = double(yHat(k)~=targs(k));
    catch e
        disp(['Graph ',num2str(k),': ',e.message]);
        outStruct(k).yhat = nan;
        outStruct(k).lhat = nan;
    end
    
    
    % Return a big struct
    outStruct(k).lat0 = lat0;
    outStruct(k).lat1 = lat1;
    outStruct(k).p0 = p0;
    outStruct(k).p1 = p1;
    outStruct(k).kCluster = kCluster;
    outStruct(k).latDim = d;
    outStruct(k).test = k;
    outStruct(k).targs = targs(k);
    
   
end

% % This plot will show a graph log likelihood ratios
% figure(303);
% clf; hold all;
% plot( all_ind.y0trn, logProb0(all_ind.y0trn)-logProb1(all_ind.y0trn),'r.');
% plot( all_ind.y1trn, logProb0(all_ind.y1trn)-logProb1(all_ind.y1trn),'b.');

disp(['Clusters:',num2str(kCluster),', LatD:',num2str(d),', Lhat=',num2str(mean([outStruct.lhat]))]);


end