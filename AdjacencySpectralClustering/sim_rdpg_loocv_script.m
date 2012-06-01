clear, clc, clf

n=70;

Z0 = [repmat([.8,.2,.2],23,1);repmat([.2,.8,.2],25,1);repmat([.2,.2,.8],22,1)];
Z1 = [repmat([.8,.2,.2],23,1);repmat([.2,.8,.2],25,1);repmat([.2,.2,.6],22,1)];

nG=50;
class0 = 1:(nG/2);
class1 = (nG/2+1):nG;
As = zeros(n,n,nG);

% Generate random As based on feature vectors
for i=1:(n-1)
    for j=(i+1):n
        for k=class0
            As(i,j,k) = double(rand<(Z0(i,:)*Z0(j,:)'));
            As(j,i,k) = As(i,j,k);
        end
        for k=class1
            As(i,j,k) = double(rand<(Z1(i,:)*Z1(j,:)'));
            As(j,i,k) = As(i,j,k);
       end
    end
end
clear i j k

targs = [zeros(1,length(class0)),ones(1,length(class1))];


d=10;
lat = zeros(n,d,nG);

%Compute Latent Features Class-wise
lat(:,:,class0) = estimate_latent_features_eig(As(:,:,class0), d);
lat(:,:,class1) = estimate_latent_features_eig(As(:,:,class1), d);

% Compute Clusters
kCluster = 3;
T = clusterdata(lat(:,:,class0(1)),kCluster);
[cluster_idx0, cluster_centroid0] = grouped_kmeans(lat(:,:,class0),kCluster,T);
T = clusterdata(lat(:,:,class1(1)),kCluster);
[cluster_idx1, cluster_centroid1] = grouped_kmeans(lat(:,:,class1),kCluster,T);
meanLat0 = cluster_centroid0(cluster_idx0,:); 
meanLat1 = cluster_centroid1(cluster_idx1,:);

% Plot true edge probabilities and estimated edge probabilites
figure(5);
subplot(2,2,1); imagesc(Z0*Z0'); caxis([0,1]); colorbar;
subplot(2,2,2); imagesc(Z1*Z1'); caxis([0,1]); colorbar;
subplot(2,2,3); imagesc(meanLat0*meanLat0'); caxis([0,1]); colorbar;
subplot(2,2,4); imagesc(meanLat1*meanLat1'); caxis([0,1]); colorbar;

% Do LOOCV outputs struct with lots of stuff
% Lhat = clustered_lat_feat_loocv(As,targs,d,kCluster);

%%

figure(6)
A=zeros(2*n);
A(1:n,1:n)=Z0*Z0';
A(1:n,n+1:2*n)=Z0*Z1';
A(n+1:2*n,1:n)=Z1*Z0';
A(n+1:2*n,n+1:2*n)=Z1*Z1';
subplot(121), imagesc(A), colorbar

B=zeros(2*n);
B(1:n,1:n)=meanLat0*meanLat0';
B(1:n,n+1:2*n)=meanLat0*meanLat1';
B(n+1:2*n,1:n)=meanLat1*meanLat0';
B(n+1:2*n,n+1:2*n)=meanLat1*meanLat1';
subplot(122), imagesc(B), colorbar



