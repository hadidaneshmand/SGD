import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;



public class exp_script_generator {
 public static void main(String[] args) {
	if(args.length < 2 ){ 
		throw new RuntimeException("NOT ENOUGH INPUT"); 
	}
	try {
		String plainscript = args[0];
		String initexp_name = "sgd_saga_adapt_efficient"; 
		String exp_runner = "run_exp"; 
		String old_lib = ".:build/"; 
		String new_lib = ".:build/:lib/*"; 
		for(int i=1;i<args.length;i++){
			String exp_name = args[i]; 
			Scanner sc = new Scanner(new BufferedReader(new FileReader(new File(plainscript+".sh"))));
			BufferedWriter bw = new BufferedWriter(new FileWriter(new File(plainscript+"_"+exp_name+".sh"))); 
			while(sc.hasNext()){
				String line = sc.nextLine(); 
				line = line.replaceAll(initexp_name, exp_runner);
				line = line.replaceAll(old_lib, new_lib);
				line += " "+exp_name+"\n"; 
				bw.write(line);
			}
			bw.flush();
			bw.close();
		}
		
	} catch (IOException e) {
		e.printStackTrace();
	}
}
}
