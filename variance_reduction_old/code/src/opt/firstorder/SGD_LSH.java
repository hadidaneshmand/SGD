package opt.firstorder;

import opt.loss.Loss;

public class SGD_LSH extends SGD {
	
	public SGD_LSH(Loss loss, int bucketsize,int tabelsize) {
		super(loss);
		
//		
	}
	
//	@Override
//	public DataPoint getImportantGradient() {
//		DataPoint out = null; 
//		DataPoint query = new SparsePoint(); 
//		for(int i=0;i<loss.getDimension();i++){ 
//		   query.set(i, w.get(i));
//		}
//		query.set(loss.getDimension(), -1.0);
//		int[] queryhash = lsh.hash(query); 
//		ArrayList<Integer> tableIndices = new ArrayList<Integer>(); 
//		for(int i=0;i<tabelsize;i++){ 
//			tableIndices.add(i); 
//		}
//		Collections.shuffle(tableIndices, new Random());
//		int index = -1; 
//		for(int i=0;i<tabelsize;i++){ 
//			int randtable_index = tableIndices.get(i); 
//			Hashtable<Integer, Integer> r_table = tables.get(randtable_index); 
//			if(r_table.get(queryhash[i])!= null){ 
//				
//			}
//			
//		}
//		return super.getImportantGradient();
//	}

}
