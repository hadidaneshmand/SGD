package opt;

import java.util.ArrayList;
import java.util.Calendar;
import java.util.Collections;
import java.util.Random;

public class Adapt_Strategy {
	
	protected int subsamplesi = 0; 
	protected ArrayList<Integer> indices; 
	protected int T = 0; 
	private boolean doubling;
	protected int totalSize; 
    
    public ArrayList<Integer> getSubInd(){ 
    	ArrayList<Integer> out = new ArrayList<Integer>(); 
    	for(int i=0;i<subsamplesi;i++){ 
    		out.add(indices.get(i));
    	}
    	return out;
    }
	public Adapt_Strategy(int totalSize, int subsamplesize,boolean doubling) {
		T = 0; 
		this.totalSize = totalSize; 
		setSubsamplesi(Math.min(subsamplesize,totalSize)); 
		indices = new ArrayList<Integer>();
		for(int i=0;i<totalSize;i++){ 
			indices.add(i);
		}
		Collections.shuffle(indices);
		this.setDoubling(doubling);
	}

	public int Tack(){ 
		T++; 
		Random random = new Random();
		random.setSeed(Calendar.getInstance().getTimeInMillis()); 
		int index = random.nextInt(getSubsamplesi());
		if(getSubsamplesi()<totalSize-1 && T>2*getSubsamplesi()){ 
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

	public int getSubsamplesi() {
		return subsamplesi;
	}

	public void setSubsamplesi(int subsamplesi) {
		this.subsamplesi = subsamplesi;
	}
	public Adapt_Strategy clone_strategy(){ 
		Adapt_Strategy out = new Adapt_Strategy(totalSize, subsamplesi, isDoubling());
		out.T = T; 
		out.indices = (ArrayList<Integer>) indices.clone(); 
		return out; 
	}
	public boolean isDoubling() {
		return doubling;
	}
	public void setDoubling(boolean doubling) {
		this.doubling = doubling;
	}
	

}
