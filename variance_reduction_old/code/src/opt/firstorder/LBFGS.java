package opt.firstorder;

import java.util.LinkedList;

import data.DataPoint;
import opt.Adapt_Strategy;
import opt.loss.Adaptss_loss_efficient;
import opt.loss.Loss;
import sun.awt.image.ImageWatched.Link;

public class LBFGS extends FirstOrderOpt {
	
	LinkedList<DataPoint> ys;
	LinkedList<DataPoint> ss; 
	LinkedList<Double> rhos; 
	int m; 
	int em; 
	double c_1 = 0.0001;
	double c_2 = 0.97;
	double beta = 0.9; 
	int max_itr = 20;
    boolean finished = false; 
	public LBFGS(Loss loss,int m) {
		super(loss);
		this.m = m; 
		ys = new LinkedList<DataPoint>(); 
		ss = new LinkedList<DataPoint>(); 
		rhos = new LinkedList<Double>(); 
		em = 0; 
	}
	public double wolfe_condition_step(DataPoint p_k, DataPoint gradient){ 
		double alpha = 0.9; 
		DataPoint w_new = (DataPoint) w.add(p_k.multiply(alpha)); 
	    double f_new = loss.getLoss(w_new);
	    DataPoint g_new = loss.getAverageGradient(w_new);
	    double f = loss.getLoss(w); 
	    
	    double condition_II_one_side =c_2*p_k.scalarProduct(gradient);
	    int c = 0; 
//	    System.out.println("==========check wolf conditions=============");
//	    System.out.println("condition_II_one_side:"+condition_II_one_side+",otherside:"+p_k.scalarProduct(g_new) );
	    while(true){
	    	if(f_new <= f+c_1*alpha*gradient.scalarProduct(p_k) && p_k.scalarProduct(g_new) >= condition_II_one_side){
	    		break; 
	    	}
	    	if(c>max_itr){ 
	    		return -1; 
	    	}
	    	c++;
	    	System.out.println("======iteration:"+c+",alpha:"+alpha);
	    	alpha = alpha*beta; 
	    	w_new = (DataPoint) w.add(p_k.multiply(alpha)); 
	    	f_new = loss.getLoss(w_new);
	    	g_new = loss.getAverageGradient(w_new);
	    	System.out.println("f:"+f);
	    	System.out.println("fnew:"+f_new+",otherside:"+(f+c_1*alpha*gradient.scalarProduct(p_k)));
	    	System.out.println("condition_II_one_side:"+condition_II_one_side+",otherside:"+p_k.scalarProduct(g_new) );
	    }
	    return alpha;
	}
	public void iterate_once(){
//		L-BFGS two-loop recursion: check numerical optimization page 178-179
		num_computed_gradients+= loss.getDataSize(); 
		if(finished){ 
			return;
		}
		DataPoint q = loss.getAverageGradient(w); 
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
		DataPoint g = loss.getAverageGradient(w);
		double alpha = wolfe_condition_step(p,g);// Wolfe conditions
		if(alpha<0){ 
			finished = true; 
			return;
		}
		DataPoint w_new = (DataPoint) w.add(p.multiply(alpha)); 
		DataPoint g_new = loss.getAverageGradient(w_new);
		 
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
	}
	@Override
	public void Iterate(int stepNum) {
		for(int i=0;i<stepNum;i++){
			iterate_once();
		}
	}

	@Override
	public String getName() {
		if(loss instanceof Adaptss_loss_efficient){ 
			return "dyna-lbfgs"; 
		}
		return "lbfgs";
	}

	@Override
	public FirstOrderOpt clone_method() {
		LBFGS lbfgs = new LBFGS(loss.clone_loss(), m);
		lbfgs.setLearning_rate(getLearning_rate());
		lbfgs.setNum_computed_gradients(getNum_computed_gradients());
		lbfgs.setParam(this.cloneParam());
		lbfgs.ys = (LinkedList<DataPoint>) this.ys.clone(); 
		lbfgs.ss = (LinkedList<DataPoint>) this.ss.clone(); 
		lbfgs.rhos = (LinkedList<Double>) this.rhos.clone(); 
		lbfgs.beta = beta; 
		lbfgs.c_1 = c_1; 
		lbfgs.em = em; 
		return lbfgs;
	}

}
