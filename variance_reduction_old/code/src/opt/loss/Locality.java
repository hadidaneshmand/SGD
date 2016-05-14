package opt.loss;

import java.util.List;

import data.DataPoint;
import opt.SampleSizeStrategy;

public class Locality extends Dyna_samplesize_loss_e{
	double c; 
	DataPoint storedGrad  = null; 
	boolean use_stored = false; 
	Loss loss_without_lambda; 
	List<Integer> inds; 
	public Locality(Loss loss, SampleSizeStrategy as, double c) {
		super(loss, as);
		loss_without_lambda = loss.clone_loss(); 
		loss_without_lambda.set_lambda(0.0);
		set_lambda(1.0/as.getSubsamplesi());
		System.out.println("constractor->lambda:"+getLambda());
		this.c = c; 
		inds = as.getAllInds();
	}
	

	@Override
	public void tack(DataPoint w) {
		
		int newss = as.getSubsamplesi(); 
		System.out.println("sub_size:"+as.getSubInd().size());
		DataPoint sum_grad = (DataPoint) loss_without_lambda.getStochasticGradient(as.getSubInd(), w);
		System.out.println("gradient has been computed");
		DataPoint grad = sum_grad.clone_data();
		
		sum_grad = (DataPoint) sum_grad;
		System.out.println("tack starts!!");
		sum_grad = (DataPoint) sum_grad.multiply(newss);
		int incrementFactor = 0; 
		double norm = grad.squaredNorm();
		System.out.println("norm of gradient:"+norm+",samplesize:"+1.0/newss);
		long t1 = System.currentTimeMillis();
		for(int j=0;j<20;j++){ 
//			System.err.println("++++++++++++++++++++++++++++++");
			incrementFactor = (int) (newss*c);
//			System.out.println("increment factor:"+incrementFactor+",newss:"+newss);
			int increasess = newss+incrementFactor; 
			if(increasess >= loss.getDataSize()){
				increasess = loss.getDataSize()-1; 
			}
			DataPoint new_grad = loss_without_lambda.getSumOfGradient(inds.subList(newss, increasess), w);
			sum_grad = (DataPoint) sum_grad.add(new_grad);
//			System.out.println("norm of sum:"+sum_grad.getNorm());
			grad = (DataPoint) ((DataPoint) sum_grad.multiply(1.0/increasess)).add(w.multiply(1.0/increasess)); 
			norm = grad.squaredNorm();
//			System.out.println("norm:"+norm+",sample size:"+(1.0/increasess));
			if(norm > 1.0/increasess || loss.getDataSize() ==increasess){ 
				break; 
			}
			else{
			   use_stored = true; 
			   storedGrad = grad; 
			   newss += incrementFactor;
			}
		}
		long t2   = System.currentTimeMillis();
		System.out.println("TIME:"+(t2 - t1));
		System.out.println("final sample size:"+newss);
		as.setSampleSize(newss);
		loss.set_lambda(1.0/as.getSubsamplesi());
	}
	@Override
	public DataPoint getAverageGradient(DataPoint w) {
		if(use_stored){
			use_stored = false;
			return storedGrad; 
		}
		return super.getAverageGradient(w);
	}

	@Override
	public Loss clone_loss() {
		Locality out = new Locality(loss.clone_loss(), as.clone_strategy(), c); 
		out.storedGrad = this.storedGrad; 
		out.use_stored = this.use_stored; 
		return out;
	}

	@Override
	public String getType() {
		return "locality";
	}
	
	
	

}
