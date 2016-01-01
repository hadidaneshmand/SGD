%filename = '../../outs/ijcnn1_streaming_test.txt'; 
filename = 'in/covtype.txt'; 
[names, res] = HadiPlotReader(filename); 
n = length(names); 
colors = {'r','b','k','g', 'm','y'};
types = {'o','*','.','+','s','d','x',};
t = res{length(res)}; 
t = mean(t,1); 
fig = figure();
for i =1:length(res)-1
   series = res{i};     
   mean_s = mean(series,1); 
   plot(t,mean_s,'Color',colors{i},'LineWidth',2);  
   hold on; 
end
legend(names,'FontSize',11);
ylabel('$\log[|R_n(w^s)-R_n(w^*)|]$','Interpreter','latex','FontSize',13);
xlabel('Iterations','Interpreter','latex','FontSize',13);
shading interp;
set(fig,'PaperPositionMode','auto');
print(fig,'-depsc','plots/covtype');