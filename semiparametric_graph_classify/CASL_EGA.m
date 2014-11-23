%% try to get something working with CASL data

clearvars, clc,
fpath = mfilename('fullpath');
findex=strfind(fpath,'/');

%% load graphs

listing = dir('../Data/CASLDelivery_0419/');

As=nan(70,70,48);
for i=3:length(listing)-1
    temp=load([fpath(1:findex(end-1)), 'Data/CASLDelivery_0419/', listing(i).name]);
    As(:,:,i-2)=temp.fibergraph+temp.fibergraph';
end

%% load labels

labelData = importdata([fpath(1:findex(end-1)), 'Data/CASLDelivery_0419/', listing(end).name]);
Ys = labelData.data;


%% LOO loop 
ds=unique(round(logspace(0,log10(40),40)));
for d=1:length(ds)
    
    % ASE
    lat = estimate_latent_features_eig( As, ds(d));
    
    % LOL
    task.types={'DENL'};
    task.ks=1:45;
    Yhat=nan(length(Ys),length(task.ks));
    for i=1:length(Ys)
        lat2=reshape(lat,[70*ds(d), 48]);
        
        sample=lat2(:,i);
        training=lat2;
        training(:,i)=[];
        
        group=Ys;
        group(i)=[];
        
        LOL_Out = LOL_classify(sample',training',group,task);
        Yhat(i,:)=LOL_Out{1};
    end
    
    for k=1:length(task.ks)
        err(d,k)=sum(Yhat(:,k) == Ys)/length(Ys);
    end
end

%%
imagesc(err)
[min_err, min_ind] = min(err(:));
[Dmin,Kmin] = ind2sub([length(ds),length(Ys)], min_ind);