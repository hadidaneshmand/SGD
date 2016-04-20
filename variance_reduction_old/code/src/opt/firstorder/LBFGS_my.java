package opt.firstorder;

import java.util.LinkedList;

import data.DataPoint;
import opt.Adapt_Strategy;
import opt.loss.Dyna_samplesize_loss_e;
import opt.loss.Loss;
import opt.loss.adaptive_loss;
import sun.awt.image.ImageWatched.Link;
import sun.swing.BakedArrayList;

public class LBFGS_my extends FirstOrderOpt {
	
	LinkedList<DataPoint> ys;
	LinkedList<DataPoint> ss; 
	LinkedList<Double> rhos; 
	int m; 
	int em; 
	
    boolean finished = false; 
	public LBFGS_my(Loss loss,int m) {
		super(loss);
		this.m = m; 
		ys = new LinkedList<DataPoint>(); 
		ss = new LinkedList<DataPoint>(); 
		rhos = new LinkedList<Double>(); 
		em = 0; 
	}
	
	public void iterate_once(){
//		L-BFGS two-loop recursion: check numerical optimization page 178-179
		System.out.println("datasize:"+getLoss().getDataSize());
		num_computed_gradients+= getLoss().getDataSize(); 
		if(finished){ 
			return;
		}
		DataPoint q = getLoss().getAverageGradient(w); 
		double[] alphas = new double[em];  
		for(int i=em-1;i>-1;i--){ 
			alphas[i] = rhos.get(i)*ss.get(i).scalarProduct(q);
			q = (DataPoint) q.add(ys.get(i).multiply(-1.0*alphas[i])); 
		}
		double gamma = 1; 
		if(em>0){ 
			gamma = ss.getLast().scalarProduct(ys.getLast())*1.0/ys.getLast().scalarProduct(ys.getLast()); 
		}
		DataPoint r = (DataPoint) q.multiply(gamma); 
		for(int i=0;i<em;i++){
			double beta = rhos.get(i)*ys.get(i).scalarProduct(r); 
			r = (DataPoint) r.add(ss.get(i).multiply(alphas[i]-beta));
		}
		DataPoint p = (DataPoint) r.multiply(-1.0); 
		DataPoint g = getLoss().getAverageGradient(w);
//		double alpha = wolfe_condition_step(p,g);// Wolfe conditions
		double alpha = backtracking_line_search(p); 
		System.out.println("alpha:"+alpha);
//		if(num_computed_gradients/(1.0*loss.getDataSize())>30){ 
//			alpha = 1.5; 
//		}
		if(alpha<0){ 
			finished = true; 
			return;
		}
		DataPoint w_new = (DataPoint) w.add(p.multiply(alpha)); 
		DataPoint g_new = getLoss().getAverageGradient(w_new);
		 
		DataPoint y_k = (DataPoint) g_new.subtract(g);
		DataPoint x_k = (DataPoint) w_new.subtract(w);
		ys.addLast(y_k);
		ss.addLast(x_k);
		rhos.addLast(1.0/y_k.scalarProduct(x_k));
		if(em<m){ 
			  em++; 
	     }
		else{
			ys.pollFirst();
			ss.pollFirst(); 
			rhos.pollFirst(); 
		}
		w = w_new;
		if(loss instanceof adaptive_loss){ 
			((adaptive_loss) loss).tack(); 
		}
	}
	@Override
	public void Iterate(int stepNum) {
		for(int i=0;i<stepNum;i++){
			iterate_once();
		}
	}

	@Override
	public void setName() {
		name = getLoss().getType()+"-lbfgs"; 
	}

	@Override
	public FirstOrderOpt clone_method() {
		LBFGS_my lbfgs = new LBFGS_my(getLoss().clone_loss(), m);
		lbfgs.setStepSize(getStepSize());
		lbfgs.setNum_computed_gradients(getNum_computed_gradients());
		lbfgs.setParam(this.clone_w());
		lbfgs.ys = (LinkedList<DataPoint>) this.ys.clone(); 
		lbfgs.ss = (LinkedList<DataPoint>) this.ss.clone(); 
		lbfgs.rhos = (LinkedList<Double>) this.rhos.clone(); 
		lbfgs.setBeta(beta); 
		lbfgs.setC_1(c_1); 
		lbfgs.em = em; 
		return lbfgs;
	}

}
