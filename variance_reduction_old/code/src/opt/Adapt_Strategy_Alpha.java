package opt;

import java.util.ArrayList;
import java.util.Random;

public class Adapt_Strategy_Alpha extends Adapt_Strategy {
    private double alpha; 
	public Adapt_Strategy_Alpha(int totalSize, int subsamplesize,
			boolean doubling,double alpha) {
		super(totalSize, subsamplesize, doubling);
		this.setAlpha(alpha);
	}
	@Override
	public int Tack() {
        T++; 
		int index = utils.getInstance().getGenerator().nextInt(getSubsamplesi());
		if(getSubsamplesi()<totalSize-1){ 
			double r = utils.getInstance().r.nextDouble();
			if(r<getAlpha()){
				setSubsamplesi(Math.min(1+getSubsamplesi(),totalSize-1)); 
			}
//			System.out.println("T="+T);
		}
//		System.out.println("T:"+T+",S:"+getSubsamplesi()+",index:"+index);
		index = indices.get(index); 
		return index;
	}
	@Override
	public Adapt_Strategy clone_strategy() {
		Adapt_Strategy_Alpha out = new Adapt_Strategy_Alpha(totalSize, subsamplesi, isDoubling(),getAlpha());
		out.T = T; 
		out.indices = (ArrayList<Integer>) indices.clone(); 
		return out; 
	}
	public double getAlpha() {
		return alpha;
	}
	public void setAlpha(double alpha) {
		this.alpha = alpha;
	}
}
