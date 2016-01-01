package opt.firstorder;


import opt.Adapt_Strategy;
import opt.loss.Loss;
import data.DataPoint;
import data.DensePoint;

public class SAGA_Adapt extends SAGA {
	Adapt_Strategy as; 
    protected double mu; 
    protected double L; 
    @Override
    public String getName() {
    	if(as.isDoubling()){
    		return "ADAPTDoubling";
    	}
    	return "ADAPTSAGA";
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
			setLearning_rate(0.3/(L+as.getSubsamplesi()*mu));
			// Compute stochastic gradient for p
			DataPoint gp = loss.getStochasticGradient(index, w);
			// Compute SAGA gradient
			DataPoint g = VarianceCorrection( index, gp);
			updateMemory(gp, index);
			// gradient step
			w = (DataPoint) w.subtract(g.multiply(getLearning_rate()));
							
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
		SAGA_Adapt out = new SAGA_Adapt(loss.clone_loss(), as.clone_strategy(),mu,L);
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
