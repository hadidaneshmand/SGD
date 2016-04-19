package opt.firstorder;


import opt.Adapt_Strategy;
import opt.Adapt_Strategy_Alpha;
import opt.Adapt_Strategy_iid;
import opt.loss.Loss;
import data.DataPoint;
import data.DensePoint;
import data.DensePoint_efficient;

public class SAGA_Adapt extends SAGA {
	Adapt_Strategy as; 
    protected double mu; 
    protected double L; 
    @Override
    public void setName() {
    	if(as.isDoubling()){
    		name = "dyna-alternating";
    	}
    	if(as instanceof Adapt_Strategy_iid){ 
    		name = "dyna-linear";
    	}
    	if(as instanceof Adapt_Strategy_Alpha){
    		name = "dyna-"+((Adapt_Strategy_Alpha)as).getAlpha();   
    	}
    	name = "dyna-SAGA";
    }
   
	public SAGA_Adapt(Loss loss,Adapt_Strategy as, double mu, double L) {
		super(loss, as.getSubsamplesi());
		this.as = as; 
		this.mu = mu; 
		this.L = L; 
	}
	@Override
	public void Iterate(int stepNum) {
		for (int i = 0; i < stepNum; ++i) {
			int index = as.Tack();
//			System.out.println("samplesize:"+as.getSubsamplesi()+",index:"+index);
			setStepSize(0.3/(L+as.getSubsamplesi()*mu));
			// Compute stochastic gradient for p
			DataPoint gp = getLoss().getStochasticGradient(index, w);
			// Compute SAGA gradient
			DataPoint g = VarianceCorrection( index, gp);
			updateMemory(gp, index);
			// gradient step
			w = (DataPoint) w.subtract(g.multiply(getStepSize()));
							
		}

	}
	public double getMu() {
		return mu;
	}
	public void setMu(double mu) {
		this.mu = mu;
	}
	public double getL() {
		return L;
	}
	public void setL(double l) {
		L = l;
	}
	
	@Override
	public FirstOrderOpt clone_method() {
		SAGA_Adapt out = new SAGA_Adapt(getLoss().clone_loss(), as.clone_strategy(),mu,L);
		out.phi = new DensePoint[getLoss().getDataSize()];
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
