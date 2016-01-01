function [ names, res ] = HadiPlotReader( filename )
    fileID = fopen(filename,'r');
    while(~feof(fileID))
       si = fscanf(fileID,'%d\n',1); 
       names = cell(si,1); 
       res = cell(si,1); 
       for i=1:si
         names{i} = fscanf(fileID,'%s\n',1);
         r = fscanf(fileID,'%d\n',1); 
         d = fscanf(fileID,'%d\n',1); 
         matrix = zeros(r,d); 
         for j=1:r
             for k=1:d
                 matrix(j,k) = fscanf(fileID,' %f',1); 
             end
             fscanf(fileID,'\n%d\n',1); 
         end
         res{i} = matrix;
       end
    end
end

