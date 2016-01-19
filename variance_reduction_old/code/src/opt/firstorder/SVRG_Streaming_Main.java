package opt.firstorder;

import java.util.ArrayList;
import java.util.Calendar;
import java.util.Random;

import data.DataPoint;
import data.DensePoint;
import opt.utils;
import opt.loss.Loss;

public class SVRG_Streaming_Main extends FirstOrderOpt{
	int T; 
	DataPoint avg;
	int samplesize; 
	int b;
	int state;
	int m; 
	int m_hat; 
	DataPoint past_w; 
	
	public void updatestate(){ 
		state = (state +1) % 2; 
		T = 0; 
		if(state == 0){
			System.out.println("new sample size:"+samplesize);
		}
	}
	public SVRG_Streaming_Main(Loss loss,double learning_rate,int samplesize, int b,int m) {
		super(loss);
		setLearning_rate(learning_rate);
		this.samplesize = samplesize;
		avg = DensePoint.zero(loss.getDimension());
		T = 0; 
		this.b = b; 
		state = 0; 
		this.m = m; 
	}
	public void computeAvg(int size){ 
		avg = DensePoint.zero(loss.getDimension());
		for(int i=0;i<size;i++){ 
			int rind = utils.getInstance().getGenerator().nextInt(loss.getDataSize()); 
			DataPoint p = loss.getStochasticGradient(rind, w); 
			avg = (DataPoint) avg.add(p);
		}
		avg = (DataPoint) avg.multiply(1.0/size);
		past_w = cloneParam();
	}

	@Override
	public void Iterate(int stepNum) {
		for(int i=0;i<stepNum;i++){ 
			T++; 
			if(state == 0 && T+1 == samplesize){ 
				computeAvg(samplesize);
				m_hat = utils.getInstance().getGenerator().nextInt(m);
				samplesize = Math.min(b*samplesize,loss.getDataSize()); 
				updatestate();
			}
            else if(state == 1 && T <= m_hat){ 
            	int rind = utils.getInstance().getGenerator().nextInt(loss.getDataSize()); 
            	DataPoint p = loss.getStochasticGradient(rind, w); 
    			p = (DataPoint) p.subtract(loss.getStochasticGradient(rind,past_w));
    			p = (DataPoint) p.add(avg);
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
		return "MainSSVRG";
	}

	@Override
	public FirstOrderOpt clone_method() {
		SVRG_Streaming_Main out = new SVRG_Streaming_Main(loss.clone_loss(), getLearning_rate(), samplesize, b, m); 
		if(past_w != null){
			out.past_w = new DensePoint(loss.getDimension());
			for(int i=0;i<loss.getDimension();i++){ 
				out.past_w.set(i, past_w.get(i));
			}
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
