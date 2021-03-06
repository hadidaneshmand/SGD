package opt.firstorder;

import java.util.LinkedList;
import java.util.Random;

import opt.loss.Loss;
import data.DataPoint;
import data.DensePoint;
import data.DensePoint_efficient;

public class SAGAMemo extends SAGA {
	LinkedList<Integer> memoinds;
	int memosize;
	public SAGAMemo(Loss loss, double learning_rate,int memosize) {
		super(loss, learning_rate);
		Random r = new Random(); 
		this.memosize = memosize;
		memoinds= new LinkedList<Integer>();
	}
	@Override
	public DataPoint VarianceCorrection(int index, DataPoint gp) {
		DataPoint g = gp;
		if( phi[index] != null) {
			g = (DataPoint) g.subtract(phi[index]);
			// subtract average over phi_j
		}
		g = (DataPoint) g.add(avg_phi.multiply(1.0/getLoss().getDataSize()));
		return g;
	}
	
	@Override
	public void updateMemory(DataPoint stochasticGradient, int index) {
		if(phi[index] != null) {
			// update average phi gradient
			avg_phi = (DensePoint_efficient) avg_phi.subtract(phi[index]	);
			
		} else {
			// increment number of gradients
			memoinds.push(index);
			++nGradients; 
		}
		// update average phi gradient
		avg_phi = (DensePoint_efficient) avg_phi.add(stochasticGradient);
		// store gradient in table phi
		phi[index] = stochasticGradient;
		if(memoinds.size()>memosize){
			int find = memoinds.pollLast();
//			System.out.println("phi[find]:"+(phi[find])+",find:"+find+",size:"+memoinds.size()); 
			avg_phi = (DensePoint_efficient) avg_phi.subtract(phi[find]); 
			phi[find] = null; 
			nGradients-=1;
		}
	}


	@Override
	public void setName() {
		name = "SAGAMEMO"; 
	}
	
	@Override
	public FirstOrderOpt clone_method() {
		SAGAMemo out = new SAGAMemo(getLoss().clone_loss(), getStepSize(),memosize);
		out.phi = new DensePoint[getLoss().getDataSize()];
		for(int i=0;i<getLoss().getDataSize();i++){
			out.phi[i] = phi[i];
		}
		out.avg_phi = new DensePoint_efficient(getLoss().getDimension());
		for(int i=0;i<getLoss().getDimension();i++){ 
			out.avg_phi.set(i, avg_phi.get(i));
		}
		out.w = clone_w(); 
		out.memoinds = (LinkedList<Integer>) memoinds.clone();
		return out;
	}

}
