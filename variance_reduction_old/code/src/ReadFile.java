import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;


public class ReadFile {
	public static void main(String[] args) {
		String filename = null;
		if(args.length > 0) {
			filename = args[0];
		}
		else{
			System.out.println("Config file is missed");
			return; 
		}
		try {
			BufferedReader br = new BufferedReader(new FileReader(new File(filename)));
			for(int i=0;i<20;i++){
				System.out.println(br.readLine());
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} 
	}
}
