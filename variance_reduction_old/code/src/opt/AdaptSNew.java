package opt;

import java.util.Random;

public class AdaptSNew extends Adapt_Strategy {
	boolean isFirstPhase; 
	public AdaptSNew(int totalSize, int subsamplesize, boolean doubling) {
		super(totalSize, subsamplesize, doubling);
		isFirstPhase = true; 
		System.out.println("initSize:"+subsamplesize);
	}
	public int Tack(){ 
		T++; 
		int index = (new Random()).nextInt(getSubsamplesi());
		if(isFirstPhase){
////			System.out.println("initial size:"+2*Math.log(getSubsamplesi()+1)*(getSubsamplesi()+1));
			if(getSubsamplesi()<totalSize-1 && T>Math.log(getSubsamplesi()+1)*(getSubsamplesi()+1)){
				setSubsamplesi(Math.min(1+getSubsamplesi(),totalSize-1));
				isFirstPhase = false;
			}
		}
		else{
			if(getSubsamplesi()<totalSize-1 && T>2*getSubsamplesi()){ 
				setSubsamplesi(Math.min(1+getSubsamplesi(),totalSize-1)); 
			}
		}
		
//		System.out.println("T:"+T+",S:"+getSubsamplesi()+",index:"+index);

		index = indices.get(index); 
		
		return index;
	}

}
