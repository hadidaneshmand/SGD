package opt.firstorder;

import data.DataPoint;
import data.DensePoint_efficient;
import opt.loss.Loss;

public class Accelarted extends FirstOrderOpt {
	Accelarable acc; 
	DataPoint pasty; 
	DataPoint pastx; 
	double L; 
	public Accelarted(Loss loss,Accelarable c) {
		super(loss);
		acc = c;
		pasty = DensePoint_efficient.zero(loss.getDimension()); 
		pastx = DensePoint_efficient.zero(loss.getDimension()); 
		L = loss.getMaxNorm(); 
	}

	@Override
	public void setName() {
		this.name = "accelarted_method"; 
	}
	
	
	public void iterate_once(){ 
		System.out.println("L:"+L);
		DataPoint curx = (DataPoint) pasty.add(loss.getAverageGradient(pasty).multiply(-1.0/L)); 
		double sq = Math.sqrt(L/loss.getLambda()); 
		double factor = (sq-1)/(sq+1); 
		System.out.println("factor:"+factor);
		DataPoint cury = (DataPoint) curx.multiply(factor+1); 
		cury = (DataPoint) cury.add(pastx.multiply(-1.0*factor)); 
	    pastx = curx.clone_data(); 
	    pasty = cury.clone_data(); 
	}

	@Override
	public FirstOrderOpt clone_method() {
		Accelarted out = new Accelarted(loss.clone_loss(), acc); 
		out.pastx = pastx.clone_data(); 
		out.pasty = pasty.clone_data(); 
		out.name = this.name; 
		out.w = w.clone_data(); 
		out.num_computed_gradients = num_computed_gradients; 
		return out;
	}
	@Override
	public DataPoint getParam() {
		return pastx;
	}

	
	
	

}
