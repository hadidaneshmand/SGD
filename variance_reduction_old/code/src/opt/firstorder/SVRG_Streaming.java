package opt.firstorder;

import java.util.ArrayList;
import java.util.Random;

import data.DataPoint;
import data.DensePoint;
import opt.loss.Loss;

public class SVRG_Streaming extends FirstOrderOpt{
	int T; 
	DataPoint[] phi; 
	DataPoint avg;
	int samplesize; 
	int b;
	int state;
	int m; 
	int m_hat; 
	
	public void updatestate(){ 
		state = (state +1) % 2; 
		T = 0; 
		if(state == 0){
			System.out.println("new sample size:"+samplesize);
		}
	}
	public SVRG_Streaming(Loss loss,double learning_rate,int samplesize, int b,int m) {
		super(loss);
		setLearning_rate(learning_rate);
		this.samplesize = samplesize;
		avg = DensePoint.zero(loss.getDimension());
		phi = new DataPoint[loss.getDataSize()];
		T = 0; 
		this.b = b; 
		state = 0; 
		this.m = m; 
	}
	public void computeAvg(int size){ 
		avg = DensePoint.zero(loss.getDimension());
		Random r = new Random();
		phi = new DataPoint[loss.getDataSize()];
		for(int i=0;i<size;i++){ 
			int rind = r.nextInt(loss.getDataSize()); 
			DataPoint p = loss.getStochasticGradient(rind, w); 
			phi[rind] = p; 
			avg = (DataPoint) avg.add(p);
		}
		avg = (DataPoint) avg.multiply(1.0/size); 
	}

	@Override
	public void Iterate(int stepNum) {
		Random r = new Random(); 
		for(int i=0;i<stepNum;i++){ 
			T++; 
			if(state == 0 && T+1 == samplesize){ 
				computeAvg(samplesize);
				m_hat = r.nextInt(m);
				samplesize = Math.min(b*samplesize,loss.getDataSize()); 
				updatestate();
			}
            else if(state == 1 && T <= m_hat){ 
            	int rind = r.nextInt(loss.getDataSize()); 
//            	rind = indecis.get(rind);
            	DataPoint p = loss.getStochasticGradient(rind, w); 
    			if(phi[rind]!=null){
    				p = (DataPoint) p.subtract(phi[rind]);
    				p = (DataPoint) p.add(avg);
    			}
    			p = (DataPoint) p.multiply(-1.0*getLearning_rate());
    			w = (DataPoint) w.add(p); 
			}
			if(state == 1 && T == m_hat){
				updatestate();
			}
			
		}
	}
	

	@Override
	public String getName() {
		return "StreamingSVRG";
	}

	@Override
	public FirstOrderOpt clone_method() {
		SVRG_Streaming out = new SVRG_Streaming(loss.clone_loss(), getLearning_rate(), samplesize, b, m); 
		out.phi = new DensePoint[loss.getDataSize()];
		for(int i=0;i<loss.getDataSize();i++){
			out.phi[i] = phi[i];
		}
		out.avg = new DensePoint(loss.getDimension());
		for(int i=0;i<loss.getDimension();i++){ 
			out.avg.set(i, avg.get(i));
		}
		out.w = cloneParam();
		out.state = state; 
		out.T = T; 
		out.m = m; 
		out.m_hat = m_hat; 
		out.b = b; 
		return out;
	}
	
	
}
