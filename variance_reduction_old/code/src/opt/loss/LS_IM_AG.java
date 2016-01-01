package opt.loss;

import java.util.ArrayList;
import java.util.List;

import stat.RandomSelector;
import stat.RandomVar;
import data.DataPoint;
import data.DensePoint;

public class LS_IM_AG extends LeastSquares_Importance{

	public LS_IM_AG(List<DataPoint> data, int dimension) {
		super(data, dimension);
	}
	@Override
	public DataPoint getStochasticGradient(DataPoint w) {
		List<DataPoint> gds = getAllStochasticGradients(w);
		List<RandomVar> items = new ArrayList<RandomVar>(); 
		DataPoint query = makequerypoint(w); 
		for(int i=0;i<getDataSize();i++){
			DataPoint t = makedatahash(i); 
			double t1 = t.angle(query);
			if(t1>Math.PI*0.5){ 
				t1 = t1- Math.PI;
			}
			double t2 = t1*t1; 
			double t4 = t2*t2;
			double pi2 = Math.PI*Math.PI; 
			double exp = (1-4*t2/(pi2*1))*(1-4*t2/(pi2*9));
//			double exp = (1-4*t2/pi2)*(1-t2/(pi2*4));
//			exp = 2*exp-exp*exp; 	
			items.add(new RandomVar(i,Math.pow(Math.cos(t1/2),2))); 
		} 
		RandomSelector rs = new RandomSelector(items); 
		int index = rs.getRSample().getIndex();
		System.out.println("norm:"+gds.get(index).getNorm());
		return gds.get(index);
	}
	public DataPoint makequerypoint(DataPoint w){ 
		DataPoint query = new DensePoint(getDimension()); 
		for(int i=0;i<getDimension();i++){ 
		   query.set(i, w.get(i));
		}
		query.set(getDimension(), -1.0);
		return query;
	}
	public DataPoint makedatahash(int index){ 
		DataPoint out = new DensePoint(getDimension()); 
		for(int i=0;i<getDimension();i++){ 
			out.set(i, data.get(index).get(i));
		}
		out.set(getDimension(), data.get(index).getLabel());
		return out;
	}
}
