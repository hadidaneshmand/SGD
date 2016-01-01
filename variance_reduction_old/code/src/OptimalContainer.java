import java.util.ArrayList;
import java.util.List;

import data.DataPoint;
import data.DensePoint;


public class OptimalContainer {
	List<Integer> indices; // training indices of subsample
	int m; 
	DataPoint optimalValue;
	int d; 
	public OptimalContainer(int m,int d) {
		this.m = m; 
		this.d = d; 
		optimalValue = new DensePoint(); 
		indices = new ArrayList<Integer>(); 
	}
	public void addIndex(int i){ 
		this.indices.add(i); 
	}
	public void setOptimalValue(DataPoint in){ 
		this.optimalValue = in; 
	}
	@Override
	public String toString() {
		String out = ""; 
		out += m + "\n"; 
		for(int i = 0;i<indices.size();i++){ 
			out += indices.get(i)+",";
		}
		out += "\n"; 
		out += optimalValue;
		return out;
	}
}
