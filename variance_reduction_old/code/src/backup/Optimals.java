package backup;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import data.DataPoint;
import data.DensePoint;


public class Optimals {
	public List<ArrayList<OptimalContainer>> optimal_ms;
	public OptimalContainer optimal_n;
	public List<Integer> ms; 
	public int d;
	public static Optimals ParseOptimals(String filename) {
		Optimals op = new Optimals(); 
		op.optimal_ms = new ArrayList<ArrayList<OptimalContainer>>(); 
		op.ms = new ArrayList<Integer>();
		try {
			BufferedReader br = new BufferedReader(new FileReader(filename));
			Scanner sc = new Scanner(br); 
			int d = sc.nextInt();
			op.d =d ; 
			int n = sc.nextInt(); 
			op.optimal_n = new OptimalContainer(n, d); 
			for(int i =0;i<n;i++){ 
				op.optimal_n.addIndex(sc.nextInt());
			}
			DataPoint w_n = new DensePoint(d);
			for(int i=0;i<d;i++){ 
				w_n.set(i, sc.nextDouble());
			}
			op.optimal_n.setOptimalValue(w_n);
//			System.out.println("--------------");
//			System.out.println(op.optimal_n);
			while(sc.hasNext()){
				int samplesi = sc.nextInt(); 
				op.ms.add(samplesi); 
				ArrayList<OptimalContainer> opt_ms = new ArrayList<OptimalContainer>(); 
				int numberofrepeats = sc.nextInt(); 
				for(int i=0;i<numberofrepeats;i++){
					sc.nextInt();
					OptimalContainer oc = new OptimalContainer(samplesi,d); 
					DataPoint w_m = new DensePoint(d); 
					for(int j=0;j<samplesi;j++){ 
						oc.addIndex(sc.nextInt());
					}
					for(int j=0;j<d;j++){
						w_m.set(j, sc.nextDouble());
					}
					oc.setOptimalValue(w_m);
					opt_ms.add(oc);
//					System.out.println("--------------");
//					System.out.println(oc);
				}
				op.optimal_ms.add(opt_ms);
				 
			}
		} catch (FileNotFoundException e) {
		    System.out.println("The optimal file could not been found at "+filename);
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return op;
	}
}
