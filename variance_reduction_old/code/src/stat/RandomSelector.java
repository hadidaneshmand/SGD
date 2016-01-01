package stat;

import java.util.List;
import java.util.Random;



public class RandomSelector {
    List<RandomVar> items ;
    Random rand = new Random();
    double totalSum = 0;

    public RandomSelector(List<RandomVar> items) {
    	this.items = items;
        for(RandomVar item : items) {
            totalSum = totalSum + item.getRelativeProb();
        }
    }

    public RandomVar getRSample() {

        double index = rand.nextDouble()*totalSum;
        double sum = 0;
        int i=0;
        while(sum < index ) {
             sum = sum + items.get(i++).getRelativeProb();
        }
        return items.get(Math.max(0,i-1));
    }
}