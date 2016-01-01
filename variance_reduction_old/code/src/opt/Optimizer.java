package opt;

import java.util.List;

import opt.config.Config;
import data.DataPoint;

public interface Optimizer {
	public void setConfig(Config conf);
	public void optimize(int iteration_num); 
	public void optimize(double tol); 
	public DataPoint getOptParam(); 
	public void setData(List<DataPoint> data);
	public double getObj();
}
