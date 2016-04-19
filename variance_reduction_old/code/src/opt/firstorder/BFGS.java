package opt.firstorder;

import org.ejml.simple.SimpleMatrix;

import data.DataPoint;
import opt.loss.Loss;

public class BFGS extends FirstOrderOpt{
	
	SimpleMatrix H_k = null;
	DataPoint s_k ; // x_{k+1} - x_{k} 
	DataPoint y_k; // f'_{k+1} - f'_{k} 
	public BFGS(Loss loss) {
		super(loss);
	}

	@Override
	public void setName() {
		this.name = "bfgs-"+loss.getType(); 
	}

	@Override
	public void Iterate(int stepNum) {
		for(int i=0;i<stepNum;i++){ 
			this.num_computed_gradients+=loss.getDataSize(); 
			IterateOnce();
		}
	}
	public void IterateOnce(){ 
		int d = loss.getDimension();
		if(H_k == null){ 
			H_k = SimpleMatrix.identity(loss.getDimension()); 
		}
		else{ 
			double rho_k = 1.0/y_k.scalarProduct(s_k); 
			System.out.println("rho:"+rho_k);
			System.out.println("max_h_k:"+H_k.elementMaxAbs());
			SimpleMatrix sy = s_k.crossProduct_sm(y_k, d); 
			SimpleMatrix I = SimpleMatrix.identity(d); 
			SimpleMatrix left = I.plus(sy.scale(-1.0*rho_k)); 
			SimpleMatrix right = I.plus(sy.transpose().scale(-1.0*rho_k)); 
			SimpleMatrix ss = s_k.crossProduct_sm(s_k, d); 
			H_k = left.mult(H_k).mult(right); 
			H_k = H_k.plus(ss.scale(rho_k)); 
			System.out.println("secant check:"+ s_k.squaredNormOfDifferenceTo(y_k.times(H_k)));
		}
		DataPoint gradient = loss.getAverageGradient(w); 
		DataPoint p_k = gradient.times(H_k); 
		p_k = (DataPoint) p_k.multiply(-1.0); 
		double step_size = backtracking_line_search(p_k); 
		System.out.println("step_size:"+step_size);
		s_k = (DataPoint) p_k.multiply(step_size); 
		w = (DataPoint) w.add(s_k); 
		y_k = (DataPoint) loss.getAverageGradient(w).add(gradient.multiply(-1.0));
	}

	@Override
	public FirstOrderOpt clone_method() {
		BFGS out = new BFGS(loss.clone_loss()); 
		out.setParam(this.getParam().clone_data());
		if(H_k!=null){ 
			out.s_k = this.s_k.clone_data(); 
			out.y_k = this.y_k.clone_data(); 
			out.H_k = H_k.scale(1.0);
		}
		return out;
	}

}
