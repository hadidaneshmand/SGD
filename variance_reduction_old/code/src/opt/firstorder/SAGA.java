package opt.firstorder;

import opt.loss.Loss;
import data.DataPoint;
import data.DensePoint;

public class SAGA extends VarianceReducedSG{
	protected DensePoint avg_phi;
	protected DataPoint[] phi;
	protected int nGradients;
	public SAGA(Loss loss, double learning_rate) {
		super(loss,learning_rate);
		this.setLearning_rate(learning_rate); 
		// allocate memory to store one gradient per datapoint
		phi = new DensePoint[loss.getDataSize()];
	    nGradients	 = 0; //number of gradients stored so far
		// average gradient
		avg_phi = new DensePoint(loss.getDimension());
		for(int i=0;i<loss.getDimension();i++){ 
			avg_phi.set(i, 0.0);
		}
	}

	
	@Override
	public DataPoint VarianceCorrection(int index, DataPoint gp) {
		DataPoint g = gp;
		if(phi[index] != null) {
			g = (DataPoint) g.subtract(phi[index]);
			// subtract average over phi_j
			g = (DataPoint) g.add(avg_phi.multiply(1.0/nGradients));
		}
		
		return g;
	}


	@Override
	public void updateMemory(DataPoint stochasticGradient, int index) {
		if(phi[index] != null) {
			// update average phi gradient
			avg_phi = (DensePoint) avg_phi.subtract(phi[index]	);
			
		} else {
			// increment number of gradients
			++nGradients; 
		}
		// update average phi gradient
		avg_phi = (DensePoint) avg_phi.add(stochasticGradient);
		// store gradient in table phi
		phi[index] = stochasticGradient;
		
	}


	@Override
	public String getName() {
		return "SAGA";
	}


	@Override
	public FirstOrderOpt clone_method() {
		SAGA out = new SAGA(loss.clone_loss(), getLearning_rate());
		out.phi = new DensePoint[loss.getDataSize()];
		for(int i=0;i<loss.getDataSize();i++){
			out.phi[i] = phi[i];
		}
		out.avg_phi = new DensePoint(loss.getDimension());
		for(int i=0;i<loss.getDimension();i++){ 
			out.avg_phi.set(i, avg_phi.get(i));
		}
		out.w = cloneParam(); 
		
		return out;
	}
	
}
