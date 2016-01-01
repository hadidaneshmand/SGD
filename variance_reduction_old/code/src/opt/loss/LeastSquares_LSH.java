package opt.loss;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Hashtable;
import java.util.List;
import java.util.Random;

import lsh.LSHSuperBit;
import data.DataPoint;
import data.DensePoint;

public class LeastSquares_LSH extends LeastSquares_Importance {
	ArrayList<Hashtable<Integer, Integer>> tables;
	int bucketsize; 
	int tabelsize; 
	LSHSuperBit lsh = null;
	public LeastSquares_LSH(List<DataPoint> data, int dimension, int bucketsize, int tabelsize) {
		super(data, dimension);
		try {
			lsh = new LSHSuperBit(tabelsize, bucketsize, getDimension()+1);
		} catch (Exception e) {
			e.printStackTrace();
		} 
		this.bucketsize = bucketsize; 
		this.tabelsize = tabelsize; 
		tables = new ArrayList<Hashtable<Integer,Integer>>(); 
		for(int i=0;i<tabelsize;i++){ 
			tables.add(new Hashtable<Integer, Integer>()); 
		}
		for(int i=0;i<data.size();i++){ 
			int[] hashvalues = lsh.hash(data.get(i));
			for(int j=0;j<tabelsize;j++){ 
				tables.get(j).put(hashvalues[j], i);
			}
		}
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
	
	@Override
	public DataPoint getStochasticGradient(DataPoint w) {
		
		ArrayList<Double> coffs = new ArrayList<Double>(); 
		coffs.add(1.0); 
//		coffs.add(-1.0);
		Collections.shuffle(coffs,new Random());
		ArrayList<Integer> tableIndices = new ArrayList<Integer>();
		for(int i=0;i<tabelsize;i++){ 
			tableIndices.add(i); 
		}
		Random r = new Random();
		Collections.shuffle(tableIndices, r);
		int index = -1;
		DataPoint query = makequerypoint(w);
		for(int k=0;k<coffs.size();k++){
			DataPoint query_r = (DataPoint) query.multiply(coffs.get(k));
			int[] queryhash = lsh.hash(query_r); 
			for(int i=0;i<tabelsize;i++){ 
				int randtable_index = tableIndices.get(i); 
				Hashtable<Integer, Integer> r_table = tables.get(randtable_index); 
				Integer maped = r_table.get(queryhash[i]);
				if(maped!= null){ 
					index = maped;
					break;
				}
			}
			if(index!=-1){
				break;
			}
		}
		if(index == -1){ 
			System.out.println("Failed:LSH in LSH loss");
			index = r.nextInt(getDataSize()); 
		}
		List<DataPoint> gds = getAllStochasticGradients(w);
		double avg_norm = 0; 
		for(int i=0;i<gds.size();i++){ 
			avg_norm += gds.get(i).getNorm();
		}
		avg_norm/= getDataSize();
		DataPoint g = getStochasticGradient(index, w);
		System.out.println("avg norm:"+avg_norm);
		System.out.println("norm:"+g.getNorm());
		System.out.println("-------------------");
		return g;
	}

}
