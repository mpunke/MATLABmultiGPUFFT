psi0M = randn(ny,nx,nz);
psi0M = psi0M-mean(mean(mean(psi0M)));
psi0M = psi0M+A;
v1 = zeros(ny,nx,nz);
v2 = v1;
v3 = v1;
