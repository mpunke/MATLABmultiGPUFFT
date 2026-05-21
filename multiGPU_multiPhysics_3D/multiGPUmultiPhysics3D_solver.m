%% Multiple GPU usage for Multiphysics PFC (3D)
clearvars
%4 GPUs must be available

%% intialize material and model parameters
parameters

%% initial density and velocity fields
% the density field is initialized as a random field
% velocity fields are initialized as zero
makeInitialCondition
psi = psi0M;
psiF = fftn(psi);
v1F = fftn(v1);
v2F = fftn(v2);
v3F = fftn(v3);


%% setup GPU cluster (4 GPUs)
delete(gcp('nocreate'))
GPUCluster = parcluster;
parpool(GPUCluster,4)

spmd
    gpuDevice(spmdIndex)
    reset(gpuDevice(spmdIndex))
    if spmdIndex==1
        %%initialize on GPU1
        % psi ...initial density field
        % psiF...Fourier transformed initial density field
        % v1  ...initial velocity field (first component)
        % v2  ...initial velocity field (secod component)
        % v3  ...initial velocity field (third component)
        % k   ...discretized Fourier vector (first dimension)
        % l   ...discretized Fourier vector (second dimension)
        % m   ...discretized Fourier vector (third dimension)
        kV = gpuArray(kV);
        lV = gpuArray(lV);
        mV = gpuArray(mV);
        v1 = gpuArray(v1);
        v2 = gpuArray(v2);
        v3 = gpuArray(v3);
        psiF = gpuArray(psiF);
        psi = gpuArray(psi);
    elseif spmdIndex==2
        %%initialize on GPU2
        % v1  ...initial velocity field (first component)
        % k   ...discretized Fourier vector (first dimension)
        % l   ...discretized Fourier vector (second dimension)
        % m   ...discretized Fourier vector (third dimension)
        kV = gpuArray(kV);
        lV = gpuArray(lV);
        mV = gpuArray(mV);
        v1F =gpuArray(v1F);
    elseif spmdIndex ==3
        %%initialize on GPU3
        % v2  ...initial velocity field (second component)
        % k   ...discretized Fourier vector (first dimension)
        % l   ...discretized Fourier vector (second dimension)
        % m   ...discretized Fourier vector (third dimension)
        kV = gpuArray(kV);
        lV = gpuArray(lV);
        mV = gpuArray(mV);
        v2F =gpuArray(v2F);
    elseif spmdIndex ==4
        %%initialize on GPU4
        % v3  ...initial velocity field (third component)
        % k   ...discretized Fourier vector (first dimension)
        % l   ...discretized Fourier vector (second dimension)
        % m   ...discretized Fourier vector (third dimension)
        kV = gpuArray(kV);
        lV = gpuArray(lV);
        mV = gpuArray(mV);
        v3F =gpuArray(v3F);
    end

    for s =1:tStepEuler %time iteration
        if spmdIndex==1 %GPU1 holding psi
            %semi-implicit time step update of psi
            psiF = (psiF+deltatEuler.*((kV.^2+lV.^2+mV.^2).*fftn(psi.^3) - fftn(v1.*ifftn(kV .*psiF)+v2.*ifftn(lV .*psiF)+v3.*ifftn(mV .*psiF))))./(1-deltatEuler.*...
                ((kV.^2+lV.^2+mV.^2).*((lambda-kappa)+kappa*(1+(kV.^2+lV.^2+mV.^2)).^2.*(4/3+(kV.^2+lV.^2+mV.^2)).^2)));
            psi = ifftn(psiF);

            % send psi to GPU2...GPU4
            spmdSend(psi,[2:4],2);

            % receive v1,v2,v3 from GPU2...GPU4
            v1 = spmdReceive(2,4);
            v2 = spmdReceive(3,5);
            v3 = spmdReceive(4,6);
        elseif spmdIndex==2 %GPU2 holding v1
            %receive psi from GPU1
            psi = spmdReceive(1,2);

            %fully-implicit time step update of v1
            v1F = (v1F-deltatEuler/rho0 .*(exp(1).^((1.*p)^2/2.*((kV).^2+(lV).^2+(mV).^2))).*fftn(psi.*ifftn(kV.*(fftn(psi.^3) + ...
                ((lambda-kappa)+kappa*(1+((kV).^2+(lV).^2+(mV).^2)).^2.*(4/3+((kV).^2+(lV).^2+(mV).^2)).^2)...
                .*fftn(psi)))))./(1-deltatEuler/rho0 .*...
                gammaS.*(kV.^2+lV.^2+mV.^2));
            v1 = (ifftn(v1F));

            %send v1 to GPU1
            spmdSend(v1,1,4);

        elseif spmdIndex==3 %GPU3 holding v2
            %receive psi from GPU1
            psi = spmdReceive(1,2);

            %fully-implicit time step update of v2
            v2F = (v2F-deltatEuler/rho0 .*(exp(1).^((1.*p)^2/2.*((kV).^2+(lV).^2+(mV).^2))).*fftn(psi.*ifftn(lV.*(fftn(psi.^3) + ...
                ((lambda-kappa)+kappa*(1+((kV).^2+(lV).^2+(mV).^2)).^2.*(4/3+((kV).^2+(lV).^2+(mV).^2)).^2)...
                .*fftn(psi)))))./(1-deltatEuler/rho0 .*...
                gammaS.*(kV.^2+lV.^2+mV.^2));
            v2 = (ifftn(v2F));

            %send v2 to GPU1
            spmdSend(v2,1,5);
        elseif spmdIndex==4%GPU4 holding v3
            %receive psi from GPU1
            psi = spmdReceive(1,2);

            %fully-implicit time step update of v3
            v3F = (v3F-deltatEuler/rho0 .*(exp(1).^((1.*p)^2/2.*((kV).^2+(lV).^2+(mV).^2))).*fftn(psi.*ifftn(mV.*(fftn(psi.^3) + ...
                ((lambda-kappa)+kappa*(1+((kV).^2+(lV).^2+(mV).^2)).^2.*(4/3+((kV).^2+(lV).^2+(mV).^2)).^2)...
                .*fftn(psi)))))./(1-deltatEuler/rho0 .*...
                gammaS.*(kV.^2+lV.^2+mV.^2));
            v3 = (ifftn(v3F));

            %send v3 to GPU1
            spmdSend(v3,1,6);
        end
    end

end

%% gather GPU solutions
psi = (gather(psi{1}));
v1 = (gather(v1{2}));
v2 = (gather(v2{3}));
v3 = (gather(v3{4}));

save('GPUSolution.mat','psi','v1','v2','v3')
