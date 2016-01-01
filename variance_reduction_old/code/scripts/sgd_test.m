for k = 14:14
    k
    ks(k) = k; 
    n = 2^k; 
    d = 20; 
    sigm = 0.5;
    T = 1500; 
    eta = 0.05; 
    X = randn(n,d); 
    b = rand(d,1); 
    X = normr(X); 
    y = X*b + sigm*randn(n,1); 
    w = zeros(d,1);
    wr = zeros(d,1);
    wp = zeros(d,1); 
    pi = -1; 
    losses = zeros(T,1);
    dists = zeros(T,1); 
    vars = zeros(T,1); 
    for i=1:T
        ri = randi(n);
        xi = X(ri,:); 
        yi = y(ri); 
        gi = 2*(xi*w-yi)*xi'; 
        w = w - eta*gi;
        wr = wr - eta*(gi+wp); 
        wp = gi;
        losses(1,i) = mean((X*w-y).*(X*w-y))-mean((X*b-y).*(X*b-y)); 
        losses(2,i) = mean((X*wr-y).*(X*wr-y))-mean((X*b-y).*(X*b-y));
      %  dists(i) = (w-b)'*(w-b); 
      %  T = X.*repmat((X*w-y),1,d);
       % vars(i) = sum(T(:).^2)/n; 
    end
  %  series(1,k) = mean((X*w-y).*(X*w-y));
 
  %  series(2,k) = mean((X*wr-y).*(X*wr-y));
end
plot(1:T, log2(losses(1,:)),1:T, log2(losses(2,:))); 
legend('sgd','svrg');

%   subplot(1,2,1); 
%   plot(log(losses)); 
%   subplot(1,2,2); 
%   plot(log(vars));
   b_s = inv(X'*X)*X'*y; 
   log((b_s-b)'*(b_s-b))
   
   