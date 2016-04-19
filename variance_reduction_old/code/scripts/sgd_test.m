 T = 20000;
comput_step = 100;
T2 = 2000;
ms = floor(T/comput_step)+1;
losses = zeros(4,ms);
losses_avg = zeros(4,ms);
variances = zeros(4,ms);
variances_avg = zeros(4,ms);
names = cell(4,1);
grads = zeros(4,ms);
myvar = @(X,w,y) 4*sum(sum((X.*repmat(X*w-y,1,length(w))).^2))/length(y);
dist_2_grad = @(X,w,y) sum(sum((2*X.*repmat(X*w-y,1,length(w))-repmat((2*X'*(X*w-y))',size(X,1),1)).^2))/length(y);
for sig_pow =1:4
    k = 10; 
    ks(k) = k; 
    n = 10^(4); 
    d = 20; 
    sigm = 0.5^(sig_pow);
    names{sig_pow}=sprintf('$\\sigma^2=1/2^{%i}$',sig_pow); 
    ct = 0;  
    mea = zeros(d,1); 
    mu = 0.01; 
    eta = 0.05; 
    stepsi = (1 - mu)/(d-1); 
    sigma = 0.1*eye(d);
    X = mvnrnd(mea,sigma,n);
    b = rand(d,1); 
    %X = normr(X); 
    y = X*b + sigm*randn(n,1); 
    b_n = inv(X'*X)*X'*y;
    opt_r = (X*b_n-y)'*(X*b_n-y)/n
    opt_vr =  myvar(X,b_n,y)
    l_opt_vr = log2(opt_vr)
    w = zeros(d,1);
    w_a = zeros(d,1); 
    wr = zeros(d,1);
    wp = zeros(d,1); 
    pi = -1; 
    flag1 = true; 
    flag2 = true; 
    for i=1:T
%         if( i > T2 && i<T2+4000 && flag1) 
%             eta = eta/10.0;
%             flag1 = false;
%         elseif(i>T2+4000&& flag2)
%             eta = eta*10.0; 
%             flag2 = false;
%         end
        
        ri = randi(n);
        xi = X(ri,:); 
        yi = y(ri); 
        gi = 2*(xi*w-yi)*xi'; 
        w = w - eta*gi;
        w_a = w/(i+1) + i*w_a/(i+1);
        wr = wr - eta*(gi+wp); 
        wp = gi;
        if(rem(i,comput_step) == 0)
            ct = ct + 1;
            lv = (X*w-y)'*(X*w-y)/n ;
            %lv_avg = (X*w_a-y)'*(X*w_a-y)/n - opt_r;
            lv_avg = sum((w_a-b_n).^2);
            %lv = (w-b_n)'*(w-b_n);
            losses(sig_pow,ct) = lv-opt_r; 
            losses_avg(sig_pow,ct) = lv_avg;
            variances(sig_pow,ct) = myvar(X,w,y);
            variances_avg(sig_pow,ct) = myvar(X,w_a,y); 
            grads(sig_pow,ct) = (w-b_n)'*(w-b_n) - opt_vr;
        end
      %  dists(i) = (w-b)'*(w-b); 
      %  T = X.*repmat((X*w-y),1,d);
       % vars(i) = sum(T(:).^2)/n; 
    end
  %  series(1,k) = mean((X*w-y).*(X*w-y));
 
  %  series(2,k) = mean((X*wr-y).*(X*wr-y));
end
figure();
colors = {'red','blue','green','black'};
for i=1:4
 plot( log2(losses(i,:)),'-o','color',colors{i}); 
 hold on;
end
t = 1:length(losses_avg(1,:)); 
%plot(t, log2(opt_r)*ones(1,length(t)));  
%names = {'$\eta$-SGD','optimum'}; 
legend(names,'Interpreter','latex','fontsize',16);
 ylabel('$R(w^t)$', 'Interpreter','latex','fontsize',16)
 
%  figure();
% colors = {'red','blue','green','black'};
% for i=1:1
%  plot( log2(losses_avg(i,:)),'-o','color',colors{i}); 
%  hold on;
% end
% 
%  ylabel('avg_sgd_loss', 'Interpreter','latex','fontsize',16)
%   subplot(1,2,1); 
%   plot(log(losses)); 
%   subplot(1,2,2); 
%   plot(log(vars));

% figure();
% colors = {'red','blue','green','black'};
% for i=1:4
%  plot( log2(variances_avg(i,:)),'-o','color',colors{i}); 
%  hold on;
% end
% 
% legend(names);
% ylabel('avg_variance', 'Interpreter','latex','fontsize',16)
 
%  figure();
% colors = {'red','blue','green','black'};
% for i=1:4
%  plot( log2(grads(i,:)),'color',colors{i}); 
%  hold on;
% end
% 
% legend(names);
%  ylabel('grads', 'Interpreter','latex','fontsize',16)
   