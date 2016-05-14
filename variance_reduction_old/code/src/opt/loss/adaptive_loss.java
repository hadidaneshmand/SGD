package opt.loss;

import data.DataPoint;

public interface adaptive_loss extends Loss{
	public void tack(DataPoint w); 
}
