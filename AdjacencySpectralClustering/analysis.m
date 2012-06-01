% %% Pipeline as outlined below by Jovo (via e-mail)
%
% 1) for each graph, compute the scaled sign positive graph laplacian of
% each graph (call it L_i).  one obtains this by filling in the diagonal of
% each adjacency matrix by the degree of that vertex, divided by n-1 (this
% is the expected weight of the self loop).
%
% 2) compute an svd of L_i for all i.  look at the scree plots.
% presumably, a relatively small number of eigenvalues should suffice to
% capture most of the variance.
%
% 3) build a classifier using only the first d singular vectors.  we
% haven't really thought much about this, but svd implicitly assumes
% gaussian noise, so, the optimal classifier under this assumption is a
% discriminant classifier.  the code i have in the repo for discriminant
% classifiers is still in progress, but if you want to fix it up, that'd be
% fine.  report results for the discriminant classifiers (LDA and QDA, but
% with diagonal covariance matrices and whitened covariance matrices).
%
% 4) cluster the singular vectors.  do this by initializing with a
% hierarchical clustering algorithm, and then use that to seed a k-means
% algorithm.  you can do this for a variety of values of k and d.  choose
% the one with the best BIC. marchette will send details for computing BIC
% values here, and for implementing the "k-means" algorithm, which is, in
% fact, more like a constrained finite gaussian mixture model algorithm.
%
% 5) given a particular choice of k and d, build a classifier.  here, each
% class is a mixture of gaussians, so a naive discriminant classifier is
% insufficient.  instead, one can implement a bayes plugin classifier. so,
% given the fit of the mixture distributions for the two classes, compute
% whether a test graph is closer to class 0 or class 1 distribution.

%% Setup - Change to fit your desires
clear, clc, clf

% this is just where i load stuff from my computer
lic=license;
if strcmp(lic,'636936')
    load('/Users/jovo/Research/data/MRI/BLSA/results/BLSA50_As_targs.mat');
else
    load /users/dsussman/documents/MATLAB/Brain_graphs/BLSA50_stripped.mat;
    
    As_nan = zeros([size(data(1).A),length(data)]);
    sz = size(As_nan);
    As = As_nan;
    for k=1:length(data) % replace all nan values with 0s
        As_nan(:,:,k) = data(k).A;
        for i=1:sz(1)
            for j=1:sz(2)
                if ~isnan(As_nan(i,j,k))
                    As(i,j,k) = As_nan(i,j,k);
                end
            end
        end
    end
    As_nonSym = As;
    targs = [data.y];
end
% you can set up stuff however you want here

% Symetrize the the adjacency matrix
for k = 1:length(targs)
    As(:,:,k) = As(:,:,k)+As(:,:,k)';
end

%% Step 1 & 2 - Compute Laplacians and SVDs, make scree plots
Ls = graph_laplacian(As);

[Us, Ss, Vs] = graph_svd(Ls);
sz =  size(Ss);
singVals =  zeros(sz(1),sz(3)); % col vectors of singular vals
for k=1:sz(3)
    singVals(:,k) = diag(squeeze(Ss(:,:,k)));
end

% show scree plots of for all the graphs, red is targs is true blue is
% targs is false
figure(401);
hold all;
plot( singVals(:,targs==1), 'r');
plot( singVals(:,targs==0), 'b');

%% Classify using first d singular values
% needed to add the repo to the path

% n=length(targs);
% nNode = length(singVals);
% all_ind = struct('ytrn', 1:n,'y0trn', find(targs ==0), 'y1trn', find(targs == 1));
% ind = struct('ytrn', [],'y0trn', [],'y1trn', []);
%
% discrim = struct('dLDA',1,'QDA',1);
% Lhat = struct('dLDA',cell(nNode-1-5,n),'QDA',cell(nNode-1-5,n));
% Lsem = Lhat;
% % remove each sample to do LOOCV
% for d=1:(nNode-1-5);
%     for k=1:n
%         ind.ytrn = all_ind.ytrn(all_ind.ytrn~=k);
%         ind.y0trn = all_ind.y0trn(all_ind.y0trn~=k);
%         ind.y1trn = all_ind.y1trn(all_ind.y1trn~=k);
%
%         subspace = d;
%         params = get_discriminant_params(singVals(subspace,:),...
%                                             ind,discrim);
%         [Lhat(d,k) Lsem(d,k)] = discriminant_classifiers(singVals(subspace,k),targs(k),...
%                                                 params,discrim);
%     end
% end
%
%
% Lhat_dLda = mean(reshape([Lhat.dLDA],size(Lhat)),2);
% Lhat_Qda = mean(reshape([Lhat.QDA],size(Lhat)),2);
%
% % plot Lhat vs which singular value
% figure(402); clf; plot(1:nNode-1-5,[Lhat_dLda,Lhat_Qda])
%% Estimate Latent Features using Eig
% Only need one set of Latent features because graph is undirected, so we
% use [V,D] = eig(A) where V*D*V'=A and then use low rank approximationa.

% size of the in and out latent features to be use
inSpace = [1:10];
Lhat_dLda = zeros(length(inSpace),1);
Lhat_Lda = Lhat_dLda;
Lhat_Qda = Lhat_dLda;
Lhat_dQda = Lhat_dLda;
sz = size(As);
whiten = true;
discrim = struct('LDA',1,'dLDA',1,'QDA',1,'dQDA',1);
%%
for k=1:length(inSpace)
        % get the feature vectors
        d = inSpace(k);
        
        %% All the actual work
        % [Lat_in,Lat_out] = estimate_latent_features_SVD(As,dIn,dOut);
        Lat = estimate_latent_features_eig(double(As>.49),d);
        
        features = reshape(Lat,d*sz(1),sz(3));
        % do loocv using dLDA
        Lhat =  lda_loocv(features,...
            targs,discrim,whiten);
        
        disp(['Dim: ',num2str(d),'; Lhats: ', ...
            mat2str([mean([Lhat.dLDA]),mean([Lhat.LDA]),...
                     mean([Lhat.dQDA]),mean([Lhat.QDA])])]);
                 
        %%
        
        wLhat_dLda(k) = mean([Lhat.dLDA]);
        wLhat_Lda(k) = mean([Lhat.LDA]);
        wLhat_dQda(k) = mean([Lhat.dQDA]);
        wLhat_Qda(k) = mean([Lhat.QDA]);

end
%save('/users/dsussman/documents/MATLAB/Brain_graphs/BLSA50_classified.mat');
%% Cluster features

dmax=20;
kmax=8;
cluster_Lhats = nan(4,dmax,kmax);
discrim = struct('LDA',1,'dLDA',1,'QDA',1,'dQDA',1);
for d = 1:dmax
    lat = estimate_latent_features_eig(As,d);
    
    for kCluster = 2:kmax
        disp(['Clusters=',num2str(kCluster),'; Dim=',num2str(d)]);
        n=length(targs);
        sz = size(Lat);
        idx = zeros(sz(1),1);
        cluster_idx = zeros(kCluster,1,n);
        cluster_centroid = zeros(kCluster,d,n);
        try
            [cluster_idx,cluster_centroid] = ...
                cluster_features(lat,kCluster,'distance','cosine');            
        catch e
            e
            disp('Not Enough Clusters. Results get nan');
            break;
        end
        Lhat = lda_loocv(reshape([cluster_idx,cluster_centroid],(d+1)*kCluster,n),targs,discrim,false);
        
        cluster_Lhats(:,d,kCluster) = [mean([Lhat.dLDA]),mean([Lhat.LDA]),...
            mean([Lhat.dQDA]),mean([Lhat.QDA])]';
        disp(mat2str( [mean([Lhat.dLDA]),mean([Lhat.LDA]),...
            mean([Lhat.dQDA]),mean([Lhat.QDA])]'));
    end
end
%% Make a nice plot of the first 2 dimensions of the latent features
figure;
hold all;
for k=1:5
    plot(squeeze(Lat(k,1,targs==1)),...
        squeeze(Lat(k,2,targs==1)),'.','MarkerSize',6);
end
% for k=1:kCluster
%     scatter(squeeze(cluster_centroid(k,1,targs==1)),...
%         squeeze(cluster_centroid(k,2,targs==1)),...
%         sum(cluster_idx(:,targs==1)==k));
% end
plot(cos(0:.1:(2*pi)),sin(0:.1:(2*pi)));
%%
subplot(1,2,2);
hold all;
for k=1:70
    plot(reshape(Lat(k,1,targs==0),1,sum(targs==0)),...
        reshape(Lat(k,2,targs==0),1,sum(targs==0)),'.');
end
plot(cos(0:.1:(2*pi)),sin(0:.1:(2*pi)));

%% S-T Latent featurse
sz = size(As);
d = 5;
features = zeros(d*sz(1),sz(3));
for k=1:sz(3)
    features(:,k) = reshape(scheinerman_tucker_latent_features(As(:,:,k),d),...
        d*sz(1),1);
end

Lhat = lda_loocv(features,targs,discrim,false);
[mean([Lhat.dLDA]),mean([Lhat.LDA]),...
    mean([Lhat.dQDA]),mean([Lhat.QDA])]