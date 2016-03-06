package opt;

import java.util.ArrayList;
import java.util.Collections;

public class Adapt_Strategy_Double_Full implements SampleSizeStrategy {
	int ts;// total size 
	int ss;// subsample size 
	int it;// iteration per sample size 
	int T; // tack indix
	ArrayList<Integer> indices; 
	public Adapt_Strategy_Double_Full(int totalSize, int subsamplesize, int iterationsPerSample) {
		this.ss = subsamplesize; 
		this.ts = totalSize; 
		it = iterationsPerSample; 
		indices  = new ArrayList<Integer>(); 
		for(int i=0;i<ts;i++){ 
			indices.add(i); 
		}
		Collections.shuffle(indices);
	}
	@Override
	public int Tack() {
		T++; 
		if(T>it){ 
			T = 0; 
			ss +=ss; 
			ss = Math.min(ss, ts);
		}
		return ss; 
	}
	@Override
	public SampleSizeStrategy clone_strategy() {
		Adapt_Strategy_Double_Full out = new Adapt_Strategy_Double_Full(ts,ss,it);
		out.indices= (ArrayList<Integer>) indices.clone(); 
		out.T = this.T; 
		return out;
	}
	
	
	@Override
	public ArrayList<Integer> getSubInd() {
		ArrayList<Integer> out = new ArrayList<Integer>(); 
    	for(int i=0;i<ss;i++){ 
    		out.add(indices.get(i));
    	}
    	return out;
	}
	@Override
	public int getSubsamplesi() {
		return ss;
	}
	@Override
	public String getName() {
		return "doubling";
	}

}
