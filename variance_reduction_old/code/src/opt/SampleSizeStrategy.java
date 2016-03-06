package opt;

import java.util.List;

public interface SampleSizeStrategy {
	  public List<Integer> getSubInd();
	  public int Tack();
	  public int getSubsamplesi();
	  public SampleSizeStrategy clone_strategy();
	  public String getName();
}
