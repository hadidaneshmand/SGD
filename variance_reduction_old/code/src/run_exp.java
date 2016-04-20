

public class run_exp {
	public static void main(String[] args) {
		if(args == null || args.length < 2){
			throw new RuntimeException("NOT ENOUGH INPUT"); 
		}
		for(int i=1;i<args.length;i++){
			String exp = args[i];
			switch (exp) {
			case "lambda":
				System.out.println("############# lambda test ################");
				lambda_in_obj.main(args);
				System.out.println("############# end of lambda test ############");
				break;
			case "newton":
				System.out.println("############ newton ###############");
				newton_test.main(args);
				System.out.println("########### end of newton ############");
			default:
				break;
			}
		}
		
	}
}