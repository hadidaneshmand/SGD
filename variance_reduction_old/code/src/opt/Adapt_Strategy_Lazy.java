package opt;

import java.util.ArrayList;
import java.util.Calendar;
import java.util.Collections;
import java.util.Random;

public class Adapt_Strategy_Lazy extends Adapt_Strategy{
	
	public Adapt_Strategy_Lazy(int totalSize, int subsamplesize,
			boolean doubling) {
		super(totalSize, subsamplesize, doubling);
	}
	

	public int Tack(){ 
		T++; 
		

		int index = utils.getInstance().getGenerator().nextInt(getSubsamplesi());
		if(getSubsamplesi()<totalSize-1 && T>5*getSubsamplesi()){ 
			if(isDoubling()){
				setSubsamplesi(Math.min(2*getSubsamplesi(),totalSize-1));
			}
			else{
				setSubsamplesi(Math.min(1+getSubsamplesi(),totalSize-1)); 
			}
//			System.out.println("T="+T);
			index = getSubsamplesi();
		}
//		System.out.println("T:"+T+",S:"+getSubsamplesi()+",index:"+index);

		index = indices.get(index); 
		
		return index;
	}

	
	public Adapt_Strategy_Lazy clone_strategy(){ 
		Adapt_Strategy_Lazy out = new Adapt_Strategy_Lazy(totalSize, subsamplesi, isDoubling());
		out.T = T; 
		out.indices = (ArrayList<Integer>) indices.clone(); 
		return out; 
	}
	

}
