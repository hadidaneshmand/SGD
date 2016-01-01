package opt.firstorder.old;


import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import opt.config.Config;
import data.DataPoint;
import data.SparsePoint;

public class SAGA_old extends GradientDescent {
	
	SparsePoint avg_phi;
	SparsePoint[] phi;
	int nGradients;
	
	public void TransferTabelsAndParams(SAGA_old in, ArrayList<Integer> indices){ 
		this.avg_phi = in.avg_phi; 
		for(int i=0;i<indices.size();i++){ 
			int index = indices.get(i); 
			this.phi[index] = in.phi[i]; 
		}
		this.nGradients = in.nGradients; 
		this.avg_phi = in.avg_phi; 
		this.setParam(in.getParam());
	}
	
	public SAGA_old(List<DataPoint> data, Config conf,double lambda, double eta0) {
		super(data, conf,lambda);
		setEta(eta0);
		// allocate memory to store one gradient per datapoint
		phi = new SparsePoint[data.size()];
	    nGradients	 = 0; //number of gradients stored so far
		// average gradient
		avg_phi = new SparsePoint();
	}
	public DataPoint computeStochasticGradient_SAGA(
			int index, DataPoint gp) {
		DataPoint g = gp;
		if(phi[index] != null) {
			g = (DataPoint) g.subtract(phi[index]);
			// subtract average over phi_j
			g = (DataPoint) g.add(avg_phi);
		}
		
		return g;
	}
	
	@Override
	public void optimize(int nPasses) {
		int n = data.size();
		conf.nSamplesPerPass = Math.min(conf.nSamplesPerPass, n);
		for (int k = 0; k < nPasses; ++k) {
			OneIteration(conf.nSamplesPerPass);
			double obj = computeObjective();
			obj_trace.add(obj);
			
			if(verbose){ 
				System.out.println("iteration:"+k+",obj="+obj);
			}
			
		}
	}
	public void OnePassOverData(){ 
		int n = data.size();
		conf.nSamplesPerPass = Math.min(conf.nSamplesPerPass, n);
		for (int i = 0; i < n; ++i) {

			int index = i;
			DataPoint p = (DataPoint) data.get(index);
			
			// Compute stochastic gradient for p
			DataPoint gp = computeStochasticGradient(p);
			
			// Compute SAGA gradient
			DataPoint g = computeStochasticGradient_SAGA( index, gp);
			
			
			DataPoint delta_phi = gp;
			if(phi[index] != null) {
				delta_phi = (DataPoint) delta_phi.subtract(phi[index]);
				
				// update average phi gradient
				double a = 1.0/nGradients;
				avg_phi = (SparsePoint) delta_phi.multiply(a).add(avg_phi);
									
			} else {
				// new gradient
				
				++nGradients; // increment number of gradients
				
				// update average phi gradient
				double a = 1.0/nGradients;
				double b = (1.0-a);
				avg_phi = (SparsePoint) delta_phi.multiply(a).add(avg_phi.multiply(b));
			}
			
			// store gradient in table phi
			phi[index] = (SparsePoint)gp;

			// Select step size
//						if(conf.T0 != -1) {
//							// use a decreasing step size
//							eta = conf.eta0*conf.T0/((k+1)+conf.T0);				
//						}

			// gradient step
			w = (DataPoint) w.subtract(g.multiply(eta));
							
		}
	}
	public void OneIteration(int batchsize){ 
		// create list of samples
		int n = data.size();
		Random random = new Random();
		for (int i = 0; i < batchsize; ++i) {
			
			int index = random.nextInt(n); 
			DataPoint p = (DataPoint) data.get(index);
			
			// Compute stochastic gradient for p
			DataPoint gp = computeStochasticGradient(p);
			
			// Compute SAGA gradient
			DataPoint g = computeStochasticGradient_SAGA( index, gp);
			
			
			DataPoint delta_phi = gp;
			if(phi[index] != null) {
				delta_phi = (DataPoint) delta_phi.subtract(phi[index]);
				
				// update average phi gradient
				double a = 1.0/nGradients;
				avg_phi = (SparsePoint) delta_phi.multiply(a).add(avg_phi);
									
			} else {
				// new gradient
				
				++nGradients; // increment number of gradients
				
				// update average phi gradient
				double a = 1.0/nGradients;
				double b = (1.0-a);
				avg_phi = (SparsePoint) delta_phi.multiply(a).add(avg_phi.multiply(b));
			}
			
			// store gradient in table phi
			phi[index] = (SparsePoint)gp;

			// Select step size
//						if(conf.T0 != -1) {
//							// use a decreasing step size
//							eta = conf.eta0*conf.T0/((k+1)+conf.T0);				
//						}

			// gradient step
			w = (DataPoint) w.subtract(g.multiply(eta));
							
		}
	}
	

}
