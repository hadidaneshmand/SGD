package opt.loss;

import java.util.List;

import data.DataPoint;

public class MissClass extends Logistic_Loss {

	public MissClass(List<DataPoint> data, int dimension) {
		super(data, dimension);
	}
	@Override
	public double getLoss(DataPoint w) {
		double out = 0.0; 
		for(int i=0;i<data.size();i++){ 
			DataPoint di = data.get(i);
			double y = di.getLabel(); 
			double prod = di.scalarProduct(w);
			if(y*prod<0){ 
				out++; 
			}
		}
		return out/data.size();
	}

}
