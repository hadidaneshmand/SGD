filename = '../../outs/slope_one_n.txt'; 
[names, res] = HadiPlotReader(filename); 
n = length(names);  
mat = res{2};
series = (mean(mat,1)); 
vars = (std(mat,1));
ns = res{1};
ns = ns(1,:); 
coefficients = polyfit(ns, series, 1);
fig = figure();
subplot(1,2,1);
errorbar(ns,series,vars); 
title('$\kappa = \sqrt{n}$','Interpreter','latex','FontSize',13);
ylabel('$\ln\left[|R_n(w)-R_n(w^*)|\right]$','Interpreter','latex','FontSize',13);
xlabel('$\ln(n)$','Interpreter','latex','FontSize',13);
line(ns,ns*coefficients(1)+coefficients(2),'Color','r')
legend('Suboptimality of Risk', sprintf('y = %.2f x + %.2f',coefficients(1),coefficients(2)))
filename = '../../outs/slope_half_n.txt'; 
[names, res] = HadiPlotReader(filename); 
n = length(names);  
mat = res{2};
series = (mean(mat,1)); 
vars = (std(mat,1));
ns = res{1};
ns = ns(1,:); 
coefficients = polyfit(ns, series, 1);
subplot(1,2,2);
errorbar(ns,series,vars); 
title('$\kappa = n^{0.75}$','Interpreter','latex','FontSize',13);
ylabel('$\ln\left[|R_n(w)-R_n(w^*)|\right]$','Interpreter','latex','FontSize',13);
xlabel('$\ln(n)$','Interpreter','latex','FontSize',13);
line(ns,ns*coefficients(1)+coefficients(2),'Color','r')
legend('Suboptimality of Risk', sprintf('y = %.2f x + %.2f',coefficients(1),coefficients(2)))
shading interp;
set(fig,'PaperPositionMode','auto');
print(fig,'-depsc','plots/slopes');
