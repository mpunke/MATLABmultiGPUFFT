tEnd = 1000;1E5;
deltatEuler = 1;
tStepEuler =floor(tEnd/deltatEuler);
kappa =0.459810;
epsilon =-0.1402;
lambda =kappa-epsilon;
A = 0.849;
delta = 1;
p =2.*pi.*3/sqrt(3);
ppx = p;
ppy = p;
dx = p/12;
dy = p/12;
% set domain size
nx = 1000;5E4;
ny = nx;
L1 = (nx-1).*dx;
L2 = (ny-1).*dy;
x = linspace(0,L1-dx,nx);
y = linspace(0,L2-dy,ny);
[x,y]=meshgrid(x-L1/2,y-L2/2);
kV = [0:nx-1];
lV = [0:ny-1]';


kV(floor(nx/2)+2:end) = kV(floor(nx/2)+2:end) -nx;
lV(floor(ny/2)+2:end) = lV(floor(ny/2)+2:end) -ny;

imag = 2.*pi.*1i;
kV = imag.*kV/L1;
lV = imag.*lV/L2;

lap = (kV.^2+lV.^2);