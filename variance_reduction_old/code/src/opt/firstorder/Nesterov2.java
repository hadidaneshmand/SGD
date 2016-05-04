package opt.firstorder;

import data.DataPoint;
import data.DensePoint_efficient;
import opt.loss.Loss;

public class Nesterov2 extends FirstOrderOpt {
	double pastalpha; 
	double q; 
	double L;
	DataPoint pastx; 
	DataPoint pasty; 
	Accelarable acc; 
	public Nesterov2(Loss loss, Accelarable acc_in) {
		super(loss);
		pastx = DensePoint_efficient.zero(loss.getDimension()); 
		pasty = DensePoint_efficient.zero(loss.getDimension()); 
		L = loss.getMaxNorm(); 
		q = loss.getLambda()/L; 
		pastalpha = 1.0/L; 
		acc = acc_in; 
	}
	
	@Override
	public void setName() {
		this.name = "nesterov"; 

	}

	public void iterate_once(){ 
		System.out.println("stepsize:"+step_size);
//		DataPoint newx = (DataPoint) pastx.add(acc.getGradient(pastx).multiply(-1.0*step_size));
		DataPoint newx = (DataPoint) pasty.add(acc.getGradient(pasty).multiply(-1.0*step_size));
		double y2 = Math.pow(pastalpha, 2); 
		double y4 = Math.pow(pastalpha, 4); 
		double sq = Math.sqrt(Math.pow(q, 2)-2*q*y2+ y4 +4*y2);
		double newalpha = 0.5*(-1.0*sq+q-y2);
		if(newalpha<0 || newalpha>1){ 
			newalpha = 0.5*(sq+q-y2);
		}
		if(newalpha<0 || newalpha>1){ 
			throw new RuntimeException("newalpha:"+newalpha+", which is not in range (0,1) in Nesterov2 optimization method!"); 
		}
		double beta = (pastalpha*(1-pastalpha))/(y2+newalpha);
		
		DataPoint newy = (DataPoint) newx.add(newx.add(pastx.multiply(-1.0)).multiply(beta)); 
		pastx = newx.clone_data(); 
		pasty = newy.clone_data(); 
	    pastalpha = newalpha;
	}


	@Override
	public FirstOrderOpt clone_method() {
		Nesterov2 nes = new Nesterov2(loss.clone_loss(),acc.clone_accelarable()); 
		nes.L = this.L; 
		nes.step_size = this.step_size; 
		nes.pastalpha = this.pastalpha; 
		nes.pastx = this.pastx.clone_data(); 
		nes.pasty = this.pasty.clone_data(); 
		nes.num_computed_gradients = this.num_computed_gradients; 
		return nes;
	}
	@Override
	public DataPoint getParam() {
		return pastx;
	}

}
