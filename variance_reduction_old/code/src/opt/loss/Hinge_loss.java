package opt.loss;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import data.DataPoint;
import data.DensePoint;

public class Hinge_loss extends Loss_static{
	int subsamplesi = 0; 
    boolean adaptive_sampling = false;
    ArrayList<Integer> indices; 
    int T = 0; 
    
	public Hinge_loss(List<DataPoint> data, int dimension) {
		super(data, dimension);
	}
	
	public void adaptiveSampling(int subsamplesize){ 
		adaptive_sampling = true; 
		subsamplesi = subsamplesize; 
		setLambda(1.0/(subsamplesize));
		indices = new ArrayList<Integer>();
		for(int i=0;i<getDataSize();i++){ 
			indices.add(i);
		}
		Collections.shuffle(indices);
	}
	
	public DataPoint getStochasticGradient(DataPoint w){
		if(adaptive_sampling){ 
			T++; 
			int index = (new Random()).nextInt(subsamplesi); 
			if(T<getDataSize() && T>subsamplesi){ 
				subsamplesi = Math.min(2*subsamplesi,getDataSize()); 
				setLambda(1.0/Math.sqrt(subsamplesi));
			}
			return getStochasticGradient(index, w); 
		}
		return super.getStochasticGradient(w);
	
	}

	@Override
	public DataPoint getStochasticGradient(int index, DataPoint w) {
        DataPoint g = new DensePoint(getDimension()); 
        for(int i=0;i<getDimension();i++){ 
        	g.set(i, 0);
        }
        DataPoint p = getData().get(index); 
        if(p.getLabel()*p.scalarProduct(w)<1){ 
        	g = (DataPoint) p.multiply(-1*p.getLabel());
        }
		return g;
	}

	@Override
	public double getLoss(DataPoint w) {
		double loss = 0; 
		for(int i=0;i<getDataSize();i++){
			DataPoint p = data.get(i); 
			loss+=Math.max(0, 1-p.getLabel()*w.scalarProduct(p));
		}
		loss/=getDataSize(); 
		loss+=w.squaredNorm()*lambda/2;
		return loss;
	}
	public Loss clone_loss(){ 
		Loss out = null; 
		return out;
	}

}
