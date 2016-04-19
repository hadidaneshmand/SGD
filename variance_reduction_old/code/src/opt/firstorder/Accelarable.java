package opt.firstorder;

import data.DataPoint;

public interface Accelarable {
	public DataPoint getGradient(DataPoint w);
	public double computationalComplexity(); 
	public Accelarable clone_accelarable();
}
