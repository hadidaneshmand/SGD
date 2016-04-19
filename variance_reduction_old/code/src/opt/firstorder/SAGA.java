package opt.firstorder;

import opt.loss.Loss;
import data.DataPoint;
import data.DensePoint;
import data.DensePoint_efficient;

public class SAGA extends VarianceReducedSG{
	protected DensePoint_efficient avg_phi;
	protected DataPoint[] phi;
	protected int nGradients;
	public SAGA(Loss loss, double learning_rate) {
		super(loss,learning_rate);
		this.setStepSize(learning_rate); 
		// allocate memory to store one gradient per datapoint
		phi = new DataPoint[loss.getDataSize()];
	    nGradients	 = 0; //number of gradients stored so far
		// average gradient
		avg_phi = new DensePoint_efficient(loss.getDimension());
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
			avg_phi = (DensePoint_efficient) avg_phi.subtract(phi[index]);
			
		} else {
			// increment number of gradients
			++nGradients; 
		}
		// update average phi gradient
		avg_phi = (DensePoint_efficient) avg_phi.add(stochasticGradient);
		// store gradient in table phi
//		System.out.println("phi:"+phi+",index:"+index+",datasize:"+loss.getDataSize());
		phi[index] = stochasticGradient;
		
	}


	@Override
	public void setName() {
		name = "saga"; 
	}


	@Override
	public FirstOrderOpt clone_method() {
		SAGA out = new SAGA(getLoss().clone_loss(), getStepSize());
		out.phi = new DataPoint[getLoss().getDataSize()];
		for(int i=0;i<getLoss().getDataSize();i++){
			out.phi[i] = phi[i];
		}
		out.avg_phi = new DensePoint_efficient(getLoss().getDimension());
		for(int i=0;i<getLoss().getDimension();i++){ 
			out.avg_phi.set(i, avg_phi.get(i));
		}
		out.w = clone_w(); 
		
		return out;
	}


	
	
}
