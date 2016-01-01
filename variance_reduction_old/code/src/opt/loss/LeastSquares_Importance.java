package opt.loss;

import java.util.ArrayList;
import java.util.List;

import stat.RandomSelector;
import stat.RandomVar;
import data.DataPoint;

public class LeastSquares_Importance extends LeastSquares {

	public LeastSquares_Importance(List<DataPoint> data, int dimension) {
		super(data, dimension);
	}
	@Override
	public DataPoint getStochasticGradient(DataPoint w) {
		List<DataPoint> gds = getAllStochasticGradients(w);
		List<RandomVar> items = new ArrayList<RandomVar>(); 
		for(int i=0;i<getDataSize();i++){
			items.add(new RandomVar(i,gds.get(i).getNorm())); 
		}
		RandomSelector rs = new RandomSelector(items); 
		int index = rs.getRSample().getIndex();
		return gds.get(index);
	}
	

}
