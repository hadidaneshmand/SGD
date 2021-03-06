package opt.loss;

import Jama.Matrix;
import data.DataPoint;

public class Logistic_Loss_efficient extends SecondOrderEfficientLoss {
	public static Matrix[] Hs;
	public Logistic_Loss_efficient(DataPoint[] data, int dimension) {
		super(data, dimension);
	}
	public static void buildHessians(DataPoint[] datas, int dim){ 
		Hs = new Matrix[datas.length]; 
		for(int i=0;i<datas.length;i++){
			DataPoint di = datas[i];
			Hs[i] = (di).crossProduct_sm(di, dim);
		}
	}

	@Override
	public Loss clone_loss() {
		Logistic_Loss_efficient out = new Logistic_Loss_efficient(getData(), getDimension());
		out.setLambda(lambda);
		return out;
	}

	@Override
	public DataPoint getStochasticGradient(int index, DataPoint w) {
		// loss =  log[1 + exp(−ywTx)] + (lambda/2 * ||w||^2)
		DataPoint g = null;
		DataPoint p = getData()[index]; 
		double prod = p.scalarProduct(w);
		int y = (int) p.getLabel();
		prod = Math.exp(-1*prod *y);
		g = (DataPoint) p.multiply(-1*y);
		g = (DataPoint) g.multiply(prod/(1+prod));
		g = (DataPoint) g.add(w.multiply(lambda));
		return g; 	
	}


//	@Override
//	public SimpleMatrix getHessian(DataPoint w) {
//		SimpleMatrix out = new SimpleMatrix(getDimension(),getDimension()); 
//	
//		for(int i=0;i<getDataSize();i++){ 
//			DataPoint di = getData()[i];
//			SimpleMatrix hi = (di).crossProduct_sm(di, getDimension()); 
//			double prod = -1*di.scalarProduct(w)*di.getLabel(); 
//			double g = Math.exp(prod)/(Math.pow(1+Math.exp(prod),2)); 
//			if(i == 0){
//				System.out.println("g:"+g);
//			}
//			hi = hi.scale(g);
//			out = out.plus(hi); 
//		}
//		out = out.scale(1.0/getDataSize());
//	    SimpleMatrix I = SimpleMatrix.identity(getDimension()); 
//	    I = I.scale(lambda);
//	    System.out.println("I[2,2]="+I.get(2, 2));
//	    out = out.plus(I); 
//	    System.out.println("H[2,2]="+out.get(2, 2));
//		
//		return out;
//	}

	public Matrix getHessian_exlusive_regularizer(DataPoint w, int ind) {
		DataPoint di = getData()[ind];
		Matrix hi = null; 
		if(Hs!=null){
			hi = Hs[ind];  
		}
		else { 
			hi =  (di).crossProduct_sm(di, this.getDimension());
		}
		double prod = -1*di.scalarProduct(w)*di.getLabel(); 
		double g = Math.exp(prod)/(Math.pow(1+Math.exp(prod),2)); 
		hi = hi.times(g);
		return hi;
	}
	@Override
	public double computeLoss(int index, DataPoint w) {
		DataPoint p = getData()[index];
		int y = (int) p.getLabel();
		return Math.log(1 + Math.exp(-1*y*p.scalarProduct(w)));
	}

}
