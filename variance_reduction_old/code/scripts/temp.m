%%%%%%%%%%%%%%%%%%%%%%
% Standard QR decomposition
N = 2;
A = ones(N,N+1); % N rows, N+1 columns

[qA,rA] = qr(A');

zA = qA(:,N+1)

A*zA

%%%%%%%%%%%%%%%%%%%%%%
% Run economy form on augmented matrix

N = 2;
B = zeros(N+1,N+2);
B(1:N,1:N+1) = ones(N,N+1);
B(N+1,N+2) = 1;

[qB,rB] = qr(B', 0);

zB = qB(:,N+1)

B*zB