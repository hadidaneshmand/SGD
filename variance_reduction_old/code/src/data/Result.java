package data;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Result {
	private ArrayList<String> seriesnames; 
	Map<String, List<List>> results; 
	public Result(ArrayList<String> seriesnames) {
		this.setSeriesnames(seriesnames); 
		results = new HashMap<String, List<List>>();
		for(int i=0;i<seriesnames.size();i++){ 
			results.put(seriesnames.get(i), new ArrayList<List>());
		}
		
	}
	public void addresult(String name, List exp){ 
	    List l = results.get(name); 
	    l.add(exp); 
	    results.put(name, l);
	}
	
	
	public void addresult(Result res){ 
		for(int i=0;i<res.getSeriesnames().size();i++){ 
			this.addresult(res.getSeriesnames().get(i),res.results.get(i).get(res.results.get(i).size()));
		}
	}
	public void write2File(String name){ 
		File textfile = new File(name+".txt");
			BufferedWriter bw;
			try {
				bw = new BufferedWriter(new FileWriter(textfile));
				bw.write(getSeriesnames().size()+"\n");
				for(int i=0;i<getSeriesnames().size();i++){ 
					bw.write(getSeriesnames().get(i)+"\n");
					List<List> serie = results.get((getSeriesnames().get(i)));
					bw.write(serie.size()+"\n");
					for(int j=0;j<serie.size();j++){ 
						bw.write(serie.get(j).size()+"\n");
						for(int k=0;k<serie.get(j).size();k++){
							bw.write(" "+serie.get(j).get(k));
						}
						bw.write("\n");
					}
					
				}
				bw.flush();
				bw.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
	}
	public ArrayList<String> getSeriesnames() {
		return seriesnames;
	}
	public void setSeriesnames(ArrayList<String> seriesnames) {
		this.seriesnames = seriesnames;
	}
	
}
