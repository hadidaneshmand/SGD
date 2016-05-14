package opt;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public abstract class IndexStrategy implements SampleSizeStrategy{
	protected ArrayList<Integer> inds; 
	protected int ss; 
	protected int n; 
	public IndexStrategy(int n) {
		this.n = n; 
		makeInds();
	}
	public void makeInds(){ 
		inds = new ArrayList<Integer>(); 
		for(int i=0;i<n;i++){ 
			inds.add(i); 
		}
		Collections.shuffle(inds);
	}
	 public List<Integer> getSubInd(){
		ArrayList<Integer> out = new ArrayList<Integer>(); 
		for(int i=0;i<ss;i++){
			out.add(inds.get(i)); 
		}
		return  out;
	 }
     public int getSubsamplesi(){ 
    	 return ss; 
     }
     
     @Override
    public List<Integer> getAllInds() {
    	return inds;
    }
     
    @Override
    public void setSampleSize(int ss) {
    	this.ss = ss; 
    }
	
}
