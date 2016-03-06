
public class optimal_strategies {
	public static void main(String[] args) {
		 
		int n = 800; 
		double e = 1; 
		double kappa = 8; 
		int t = (int) n;
		double rho = 1-2.0/(kappa+1); 
		double[][] u = new double[t][n]; 
		int[][] opt_n = new int[t][n];
		int[][] opt_itr = new int[t][n];
		for(int i=0;i<n;i++){
			for(int k=0;k<t;k++){
				opt_itr[k][i] = 0; 
			}
		}	
		for(int i=0;i<n;i++){ 
			u[0][i] = e; 
		}
		for(int k =0;k<t;k++){
			for(int i=2;i<n;i++){
				opt_n[k][i] = i; 
				opt_itr[k][i] = i; 
				double sub = e; 
				if(k>=i){ 
					sub = rho*u[k-i][i];
				}
				for(int j=2;j<i-1;j++){ 
					double temp = u[k][j]+ (1.0*(i-j))/(j*i);
					if(temp<sub){
						sub = temp; 
						opt_n[k][i] = opt_n[k][j]; 
						opt_itr[k][i] = opt_itr[k][j];
					}
				}
				u[k][i] = sub;
				System.out.println("u["+k+"]["+i+"]="+sub); 
			}
		}
		int m = n-1; 
		int itr = t-1;
		
		while(m>0 && itr>0){ 
		   if( opt_n[itr][m] == m){
			   System.out.println(m);
			   itr -= m; 
		   }
		   else{ 
			   System.out.println(m);
			   m = opt_n[itr][m]; 
			   itr = itr-opt_itr[itr][m]; 
		   }
		}
	}
}
