package opt.loss;

import data.DataPoint;

public class MissClass_efficient extends Logistic_Loss_efficient{

	public MissClass_efficient(DataPoint[] data, int dimension) {
		super(data, dimension);
	}
	@Override
	public double computeLoss(DataPoint w) {
		double out = 0.0; 
		for(int i=0;i<getData().length;i++){ 
			DataPoint di = getData()[i];
			double y = di.getLabel(); 
			double prod = di.scalarProduct(w);
			if(y*prod<0){ 
				out++; 
			}
		}
		return out/getData().length;
	}

}
