package opt.loss;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import data.DataPoint;
import data.DensePoint;

public class LSH_Test extends LeastSquares_Importance {
	
	public LSH_Test(List<DataPoint> data, int dimension) {
		super(data, dimension);
		
	}
	public DataPoint makequerypoint(DataPoint w){ 
		DataPoint query = new DensePoint(getDimension()); 
		for(int i=0;i<getDimension();i++){ 
		   query.set(i, w.get(i));
		}
		query.set(getDimension(), -1.0);
		return query;
	}
	public DataPoint makedatahash(DataPoint datapoint){ 
		DataPoint out = new DensePoint(getDimension()); 
		for(int i=0;i<getDimension();i++){ 
			out.set(i, datapoint.get(i));
		}
		out.set(getDimension(), datapoint.getLabel());
		return out;
	}
	public List<DataPoint> createRandomVector(int dim,int size){ 
		List<DataPoint> out = new ArrayList<DataPoint>(); 
		for(int j=0;j<size;j++){ 
			DataPoint rv = new DensePoint(getDimension()); 
			Random r = new Random();
			for(int i=0;i<dim;i++){
				rv.set(i, r.nextGaussian());
			}
			out.add(rv); 
		}
		return out;
	}
	public boolean[] getmapping(DataPoint query,List<DataPoint> rvs){ 
		boolean[] out = new boolean[rvs.size()]; 
		for(int i=0;i<rvs.size();i++){ 
			out[i] = query.scalarProduct(rvs.get(i))>0;
		}
		return out; 
	}
	@Override
	public DataPoint getStochasticGradient(DataPoint w) {
		ArrayList<Integer> indices = new ArrayList<Integer>(); 
		for(int i=0;i<getDataSize();i++){ 
			indices.add(i); 
		}
		Collections.shuffle(indices);
		
		List<DataPoint> rvs = createRandomVector(getDimension(),40);
		DataPoint query = makequerypoint(w); 
		boolean[] hashq = getmapping(query, rvs); 
		System.out.println(hashq[0]);
		int index = -1;
		for(int i=0;i<getDataSize();i++){ 
		  DataPoint p = makedatahash(data.get(indices.get(i))); 
		  boolean[] hashd = getmapping(p, rvs); 
		  boolean exp = !(getsubexp(hashd, hashq, 0) && getsubexp(hashd, hashq,  8)) ; 
		  boolean exp2 = !(getsubexp(hashd, hashq,  16) && getsubexp(hashd, hashq, 24)) ;
		  boolean exp3 =  (exp && exp2);
		  if(!exp3){ 
			  index = indices.get(i); 
			  break; 
		  }
		 
		}
		if(index == -1){ 
			System.out.println("Random!!");
			Random r = new Random(); 
			index = r.nextInt(data.size());
		}
		DataPoint g = getStochasticGradient(index,w); 
		System.out.println("norm:"+g.getNorm()+",angle:"+query.angle(makedatahash(data.get(index))));
		return g;
	}
	public boolean getsubexp(boolean[] hashd,boolean[] hashq,int index){ 
		 boolean exp1 = !( ((hashd[index] != hashq[index]) && (hashd[index+1] != hashq[index+1])));
		  boolean exp2 = !( ((hashq[index+2] != hashd[index+2]) && (hashq[index+3] != hashd[index+3])));
		  boolean exp3 = exp1 || exp2;
		  return exp3; 
	}

}
