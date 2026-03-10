B = -0.2792;
Bq = B/4;

kq1 = -sqrt(3)/2;
lq1 = -0.5;
kq2 = 0;
lq2 = 1;
kq3 = sqrt(3)/2;
lq3 = -0.5;

kq1C = -kq1;
lq1C = -lq1;
kq2C = -kq2;
lq2C = -lq2;
kq3C = -kq3;
lq3C = -lq3;

psi0M = A+Bq*(...
    exp(1i*(kq1*x  + lq1*y))+ ...
    exp(1i*(kq1C*x + lq1C*y))+ ...
    exp(1i*(kq2*x  + lq2*y))+ ...
    exp(1i*(kq2C*x + lq2C*y))+ ...
    exp(1i*(kq3*x  + lq3*y))+ ...
    exp(1i*(kq3C*x + lq3C*y)));
psi0M(abs(((x).^2+(y).^2).^0.5)>2*p)=A;