package opt;

import java.util.ArrayList;
import java.util.Collections;

public class Adapt_Strategy_Double_Full implements SampleSizeStrategy {
	int ts;// total size 
	int ss;// subsample size 
	int it;// iteration per sample size 
	int T; // tack indix
	int initial_iteration; 
	int numbIter; 
	boolean isInitialSS = true; 
	ArrayList<Integer> indices; 
	public Adapt_Strategy_Double_Full(int totalSize, int subsamplesize, int iterationsPerSample,int initial_iteration) {
		this.ss = subsamplesize; 
		this.ts = totalSize; 
		it = iterationsPerSample; 
		indices  = new ArrayList<Integer>(); 
		for(int i=0;i<ts;i++){ 
			indices.add(i); 
		}
		Collections.shuffle(indices);
		this.initial_iteration = initial_iteration; 
	}
	
	@Override
	public SampleSizeStrategy clone_strategy() {
		Adapt_Strategy_Double_Full out = new Adapt_Strategy_Double_Full(ts,ss,it,initial_iteration);
		out.indices= (ArrayList<Integer>) indices.clone(); 
		out.T = this.T; 
		out.ss = this.ss; 
		out.it = this.it; 
		out.initial_iteration = this.initial_iteration; 
		out.isInitialSS = this.isInitialSS; 
		out.ts = this.ts; 
		out.numbIter = this.numbIter; 
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

	@Override
	public int Tack() {
		System.out.println("Tack:"+numbIter);
		T++; 
		numbIter++; 
		if(isInitialSS){ 
			if(T>initial_iteration){ 
				T = 0; 
				ss += ss; 
				isInitialSS = false;
			}
		}
		else{
			if(T>=it){
				T = 0; 
				ss += ss; 
			}
		}
		ss = Math.min(ss, ts);
		return ss;
	}
	

	

}
