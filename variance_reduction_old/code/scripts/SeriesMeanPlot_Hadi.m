%  filenames = {'a9a','ijcnn_iid_included','covtype_iid_included','w8a_iid_included','rsim_iid_included','SUSY_iid_included','rcv1'};
%  outfilenames = {'a9a','ijcnn1','covtype','w8a','rsim','susy','rcv1'}; 
%  datasizes = [32561,49990,581012,49749,72309,5*10^6,20242 ];

% filenames = {'ijcnn_small_iid_included','ijcnn_medium_iid_included','rcv1_small_iid_included','rcv1_medium_iid_included','covtype_medium_iid_included','covtype_small_iid_included','w8a_small_iid_included','w8a_medium_iid_included'};
% outfilenames = {'ijcnn1_s','ijcnn1_m','rcv1_s','rcv1_m','covtype_m','covtype_s','w8a_s','w8a_m'}; 
% datasizes = [49990,49990,20242,20242,581012,581012,49749,49749];

% filenames = {'ijcnn_iid_included','rcv1_iid_included','covtype_iid_included','w8a_iid_included','SUSY_iid_included','a9a_IIDTest_','rsim_iid_included'};
% outfilenames = {'ijcnn1_iid','rcv1_iid','covtype_iid','w8a_iid','susy_iid','a9a_iid','rsim_iid'}; 
% datasizes = [49990,20242,581012,49749,5*10^6,32561,72309];
filenames = {'a9a_iid_included'};
outfilenames = {'a9a_iid_included'}; 
datasizes = [3*10^4];
for ii=1:length(filenames)
    for kk =1:2
    if(kk ==1)
        filename = sprintf('../results_arxiv/%s.txt',filenames{ii}); 
        outfilename = outfilenames{ii};
        
    end
    if(kk ==2)
        filename = sprintf('../results_arxiv/%s_test.txt',filenames{ii});
        outfilename = sprintf('%s_test',outfilename);
    end
    filename
    datasize = datasizes(ii);
    datasize = datasize*0.9
    %filename = '../../outs/covtype_5.txt'
    %filename = 'in/covtype.txt'; 
    [names, res] = HadiPlotReader(filename); 
    for i=1:length(names)
       if(strcmp(names{i},'StreamingSVRG'))
           names{i} = 'SGD/SVRG';
       end
       if(strcmp(names{i},'MainSSVRG'))
           names{i} = 'SSVRG';
       end
    end
    n = length(names); 
    colors = {'r','b','k','g', 'm','g','c',[.7 .5 0],[0.5  0    0.9],'b'};
    types = {'v-','^-','o-','+-','s-','d-','+-','->','<-','--'};
    t = res{length(res)}; 
    t = mean(t,1); 
    fig = figure();
    inds = 2:length(t); 
    if(length(t)>40 && length(t)<60)
      inds = (rem(inds,2) == 1)
    end
    if(length(t)>60 && length(t)<80)
      inds = (rem(inds,3) == 1)
    end
    if(length(t)>80 && length(t)<100)
      inds = (rem(inds,4) == 1)
    end
    if(length(t)>100 )
      inds = (rem(inds,5) == 1)
    end
    min_e = 100; 
    max_e  = -100; 
    inds_name  = zeros(n,1);
    for i =1:length(res)-1
       if(strcmp(names{i}, 'ADAPTDoubling'))
          continue; 
       end
%        if(strcmp(names{i}, 'ADAPTIID')~=1 && strcmp(names{i}, 'ADAPTSAGA')~=1)
%           continue; 
%        end
       if(strcmp(names{i}, 'ADAPTIID'))
          colors{i} = 'blue'; 
       end
       if(strcmp(names{i}, 'ADAPTIID'))
          names{i} = 'LINEAR'; 
       end
       if(strcmp(names{i},'ADAPTSAGA'))
          colors{i} = 'red'; 
       end
       inds_name(i) = true;
       if(strcmp(names{i},'ADAPTSAGA'))
          names{i} = 'ALTERNATING';
       end
       series = res{i};
       mean_s = mean(series,1);
       min_t = min(mean_s); 
       max_t = max(mean_s); 
       if(min_t<min_e)
          min_e = min_t;
       end
       if(max_t>max_e)
          max_e = max_t;
       end
       if(kk == 1)
        p = plot(t(inds),mean_s(inds),types{i},'Color',colors{i},'LineWidth',1.2,'MarkerSize',6);  
        %set(gca,'YScale','log2');
       else
        p = plot(t(inds),mean_s(inds),types{i},'Color',colors{i},'LineWidth',1.2,'MarkerSize',6);  
        %set(gca,'YScale','log2');
       end
       hold on; 
     %  set(p, 'visible', 'off');
    end
    %set(gca, 'visible', 'off');
    if(kk==1)
     vertical_y = min_e-0.5:0.01:max_e+0.5; 
     vertical_x = ones(size(vertical_y))*datasize;
    else  
     vertical_y = min_e-0.05:0.001:max_e+0.05;
     vertical_x = ones(size(vertical_y))*datasize;
    end
    p = plot(vertical_x,vertical_y,'--','Color',[.7 .5 0],'LineWidth',1.2);
    % set(p, 'visible', 'off');
    names_1 = names(inds_name>0);
    legend(names_1,'fontsize',12,'Location','northeast');
    %ylabel('$\log[|R_n(w^s)-R_n(w^*)|]$','Interpreter','latex','FontSize',13);
    %xlabel('Iterations','Interpreter','latex','FontSize',13);
    shading interp;
    set(fig,'PaperPositionMode','auto');
    length(t)
    %print(fig,'-depsc',sprintf('/Users/hadi/Documents/Education/Repositories/adapt-ss-icml2016/images/%s',outfilename));
    print(fig,'-depsc',sprintf('plots/%s',outfilename));
    end
end
%print(fig,'-depsc',sprintf('/Users/hadi/Documents/Education/Repositories/adapt-ss-icml2016/images/legend'));
%print(fig,'-depsc','plots/ijcnn_Mstar')