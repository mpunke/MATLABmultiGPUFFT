%% Single FFT with multiple GPU usage for PFC (3D)
clearvars
NGPU = gpuDeviceCount; %number of available GPUs

%% intialize material and model parameters
parameters

%% intial density
% the density field is initialized as a random field

% psi ...initial density field
% psiF...Fourier transformed initial density field
% lap ...discretized laplacian
% lin ...discretized linear operator (epsilon+L^2)nabla^2

makeInitialCondition
psiIF = psi0M;
psiF = fftn(psiIF);
lin =  (kV.^2+lV.^2+mV.^2).*((lambda-kappa)+kappa*(1+(kV.^2+lV.^2+mV.^2)).^2.*(4/3+(kV.^2+lV.^2+mV.^2)).^2);



%% prepare slab decomposition on NGPU
start = round(nx/NGPU)*([1:NGPU]-1)+1;
finish = round(nx/NGPU)*[1:NGPU];
finish(end)=nx;

for iter = 1:NGPU
    if iter==1
        start1 =start(1);
        end1   =finish(1);
    elseif iter==2
        start2 =start(2);
        end2   =finish(2);
    elseif iter==3
        start3 =start(3);
        end3   =finish(3);
    elseif iter==4
        start4 =start(4);
        end4   =finish(4);
    elseif iter==5
        start5 =start(5);
        end5   =finish(5);
    elseif iter==6
        start6 =start(6);
        end6   =finish(6);
    elseif iter==7
        start7 =start(7);
        end7   =finish(7);
    elseif iter==8
        start8 =start(8);
        end8   =finish(8);

    end
end

%% slab decomposition
% the following fields are decomposed into slabs
% psi, psiF, lap, lin

for iter = 1:NGPU
    if iter==1
        lap1 = lap(start1:end1,:,:);
        lin1 = lin(start1:end1,:,:);
        psi1 = psiIF(:,:,start1:end1);
        psiF1 =psiF(start1:end1,:,:);
    elseif iter==2
        lap2 = lap(start2:end2,:,:);
        lin2 = lin(start2:end2,:,:);
        psi2 = psiIF(:,:,start2:end2);
        psiF2 =psiF(start2:end2,:,:);
    elseif iter==3
        lap3 = lap(start3:end3,:,:);
        lin3 = lin(start3:end3,:,:);
        psi3 = psiIF(:,:,start3:end3);
        psiF3 =psiF(start3:end3,:,:);
    elseif iter==4
        lap4 = lap(start4:end4,:,:);
        lin4 = lin(start4:end4,:,:);
        psi4 = psiIF(:,:,start4:end4);
        psiF4 =psiF(start4:end4,:,:);
    elseif iter==5
        lap5 = lap(start5:end5,:,:);
        lin5 = lin(start5:end5,:,:);
        psi5 = psiIF(:,:,start5:end5);
        psiF5 =psiF(start5:end5,:,:);
    elseif iter==6
        lap6 = lap(start6:end6,:,:);
        lin6 = lin(start6:end6,:,:);
        psi6 = psiIF(:,:,start6:end6);
        psiF6 =psiF(start6:end6,:,:);
    elseif iter==7
        lap7 = lap(start7:end7,:,:);
        lin7 = lin(start7:end7,:,:);
        psi7 = psiIF(:,:,start7:end7);
        psiF7 =psiF(start7:end7,:,:);
    elseif iter==8
        lap8= lap(start8:end8,:,:);
        lin8 = lin(start8:end8,:,:);
        psi8 = psiIF(:,:,start8:end8);
        psiF8 =psiF(start8:end8,:,:);


    end
end

% save the decomposed fields
savename = append('localVars_',num2str(nx),'_NGPU_',num2str(NGPU),'.mat');
if NGPU ==2
    save(savename,'lap1','lin1','psi1','psiF1',...
        'lap2','lin2','psi2','psiF2','-v7.3')
elseif NGPU==3
    save(savename,'lap1','lin1','psi1','psiF1',...
        'lap2','lin2','psi2','psiF2',...
        'lap3','lin3','psi3','psiF3','-v7.3')
elseif NGPU==4
    save(savename,'lap1','lin1','psi1','psiF1',...
        'lap2','lin2','psi2','psiF2',...
        'lap3','lin3','psi3','psiF3',...
        'lap4','lin4','psi4','psiF4','-v7.3')
elseif NGPU==5
    save(savename,'lap1','lin1','psi1','psiF1',...
        'lap2','lin2','psi2','psiF2',...
        'lap3','lin3','psi3','psiF3',...
        'lap4','lin4','psi4','psiF4',...
        'lap5','lin5','psi5','psiF5','-v7.3')
elseif NGPU==6
    save(savename,'lap1','lin1','psi1','psiF1',...
        'lap2','lin2','psi2','psiF2',...
        'lap3','lin3','psi3','psiF3',...
        'lap4','lin4','psi4','psiF4',...
        'lap5','lin5','psi5','psiF5',...
        'lap6','lin6','psi6','psiF6','-v7.3')
elseif NGPU==7
    save(savename,'lap1','lin1','psi1','psiF1',...
        'lap2','lin2','psi2','psiF2',...
        'lap3','lin3','psi3','psiF3',...
        'lap4','lin4','psi4','psiF4',...
        'lap5','lin5','psi5','psiF5',...
        'lap6','lin6','psi6','psiF6',...
        'lap7','lin7','psi7','psiF7','-v7.3')
elseif NGPU==8
    save(savename,'lap1','lin1','psi1','psiF1',...
        'lap2','lin2','psi2','psiF2',...
        'lap3','lin3','psi3','psiF3',...
        'lap4','lin4','psi4','psiF4',...
        'lap5','lin5','psi5','psiF5',...
        'lap6','lin6','psi6','psiF6',...
        'lap7','lin7','psi7','psiF7',...
        'lap8','lin8','psi8','psiF8','-v7.3')
end

clear lap lin psiIF psiF ...
    lap1 lin1 psi1 psiF1 ...
    lap2 lin2 psi2 psiF2 ...
    lap3 lin3 psi3 psiF3 ...
    lap4 lin4 psi4 psiF4 ...
    lap5 lin5 psi5 psiF5 ...
    lap6 lin6 psi6 psiF6 ...
    lap7 lin7 psi7 psiF7 ...
    lap8 lin8 psi8 psiF8


'saved and cleared vars'

%% parallel GPU session (NGPU GPUs), on each GPU perform
% load decomposed fields in the individual GPU memory
delete(gcp('nocreate'))
GPUCluster = parcluster;
parpool(GPUCluster,NGPU)

spmd
    gpuDevice(spmdIndex)
    reset(gpuDevice(spmdIndex))
    if spmdIndex==1
        lap = gpuArray(load(savename).lap1);
        lin = gpuArray(load(savename).lin1);
        psi = gpuArray(load(savename).psi1);
        psiF = gpuArray(load(savename).psiF1);
    elseif spmdIndex ==2
        lap = gpuArray(load(savename).lap2);
        lin = gpuArray(load(savename).lin2);
        psi = gpuArray(load(savename).psi2);
        psiF = gpuArray(load(savename).psiF2);
    elseif spmdIndex ==3
        lap = gpuArray(load(savename).lap3);
        lin = gpuArray(load(savename).lin3);
        psi = gpuArray(load(savename).psi3);
        psiF = gpuArray(load(savename).psiF3);
    elseif spmdIndex ==4
        lap = gpuArray(load(savename).lap4);
        lin = gpuArray(load(savename).lin4);
        psi = gpuArray(load(savename).psi4);
        psiF = gpuArray(load(savename).psiF4);
    elseif spmdIndex ==5
        lap = gpuArray(load(savename).lap5);
        lin = gpuArray(load(savename).lin5);
        psi = gpuArray(load(savename).psi5);
        psiF = gpuArray(load(savename).psiF5);
    elseif spmdIndex ==6
        lap = gpuArray(load(savename).lap6);
        lin = gpuArray(load(savename).lin6);
        psi = gpuArray(load(savename).psi6);
        psiF = gpuArray(load(savename).psiF6);
    elseif spmdIndex ==7
        lap = gpuArray(load(savename).lap7);
        lin = gpuArray(load(savename).lin7);
        psi = gpuArray(load(savename).psi7);
        psiF = gpuArray(load(savename).psiF7);
    elseif spmdIndex ==8
        lap = gpuArray(load(savename).lap8);
        lin = gpuArray(load(savename).lin8);
        psi = gpuArray(load(savename).psi8);
        psiF = gpuArray(load(savename).psiF8);
    end

    for s =1:tStepEuler %time iteration
        %%forward fftn
        psi = fft2(psi.^3); %fft2 along first two dimensions


        %%P2P communication
        for iter = 1:NGPU %decompose each slab in a set of local chunks and
            % stack the chunks together (along third dimension)

            if iter==1
                psi1 = psi(start1:end1,:,:);
                psiTest1= spmdCat(psi1,3,1);
            elseif iter==2
                psi2 = psi(start2:end2,:,:);
                psiTest2= spmdCat(psi2,3,2);
            elseif iter==3
                psi3 = psi(start3:end3,:,:);
                psiTest3= spmdCat(psi3,3,3);
            elseif iter==4
                psi4 = psi(start4:end4,:,:);
                psiTest4= spmdCat(psi4,3,4);
            elseif iter==5
                psi5 = psi(start5:end5,:,:);
                psiTest5= spmdCat(psi5,3,5);
            elseif iter==6
                psi6 = psi(start6:end6,:,:);
                psiTest6= spmdCat(psi6,3,6);
            elseif iter==7
                psi7 = psi(start7:end7,:,:);
                psiTest7= spmdCat(psi7,3,7);
            elseif iter==8
                psi8 = psi(start8:end8,:,:);
                psiTest8= spmdCat(psi8,3,8);
            end
        end

        if spmdIndex==1
            psi=psiTest1;
        elseif spmdIndex==2
            psi=psiTest2;
        elseif spmdIndex==3
            psi=psiTest3;
        elseif spmdIndex==4
            psi=psiTest4;
        elseif spmdIndex==5
            psi=psiTest5;
        elseif spmdIndex==6
            psi=psiTest6;
        elseif spmdIndex==7
            psi=psiTest7;
        elseif spmdIndex==8
            psi=psiTest8;
        end

        psi = fft(psi,[],3); %fft along third dimension

        %%semi-implicit time step update
        psiF = (psiF+deltatEuler*lap.*psi)./(1-deltatEuler*...
            lin);
        psi = psiF;

        %%backward ifftn
        psi  = ifft(psi,[],3); %ifft along third dimension


        %%P2P communication
        for iter = 1:NGPU %decompose each slab in a set of local chunks and
            %stack the chunks together (along first dimension)
            if iter==1
                psi1 = psi(:,:,start(iter):finish(iter));
                psiTest1= spmdCat(psi1,1,1);
            elseif iter==2
                psi2 = psi(:,:,start(iter):finish(iter));
                psiTest2= spmdCat(psi2,1,2);
            elseif iter==3
                psi3 = psi(:,:,start(iter):finish(iter));
                psiTest3= spmdCat(psi3,1,3);
            elseif iter==4
                psi4 = psi(:,:,start(iter):finish(iter));
                psiTest4= spmdCat(psi4,1,4);
            elseif iter==5
                psi5 = psi(:,:,start(iter):finish(iter));
                psiTest5= spmdCat(psi5,1,5);
            elseif iter==6
                psi6 = psi(:,:,start(iter):finish(iter));
                psiTest6= spmdCat(psi6,1,6);
            elseif iter==7
                psi7 = psi(:,:,start(iter):finish(iter));
                psiTest7= spmdCat(psi7,1,7);
            elseif iter==8
                psi8 = psi(:,:,start(iter):finish(iter));
                psiTest8= spmdCat(psi8,1,8);
            end
        end


        if spmdIndex==1
            psi=psiTest1;
        elseif spmdIndex==2
            psi=psiTest2;
        elseif spmdIndex==3
            psi=psiTest3;
        elseif spmdIndex==4
            psi=psiTest4;
        elseif spmdIndex==5
            psi=psiTest5;
        elseif spmdIndex==6
            psi=psiTest6;
        elseif spmdIndex==7
            psi=psiTest7;
        elseif spmdIndex==8
            psi=psiTest8;
        end

        psi  = ifft2(psi);%ifft2 along first two dimensions
    end

end

%% stack solution
for k = 1:numel(psi)
    psiGPU{k} = gather(psi{k});
end
clear psi
psi = cat(3, psiGPU{:});
save('GPUSolution.mat','psi')






