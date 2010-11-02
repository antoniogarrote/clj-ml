package cljml;

/**
 * A SimpleStreamFilter whose logic is composed of clojure functions.
 *
 * @author Ben Mabey
 */

import clojure.lang.IFn;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.SimpleStreamFilter;
import weka.core.Capabilities.*;
import weka.core.Capabilities;

public class ClojureStreamFilter
  extends SimpleStreamFilter {

  public IFn processFn;

  public ClojureStreamFilter(IFn processFn) {
    this.processFn = processFn;
  }

  public String globalInfo() {
    return "A simple stream filter that runs a provided clojure function.";
  }

    
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();
    result.enableAllAttributes();
    result.enableAllClasses();
    result.enable(Capability.NO_CLASS);  // filter doesn't need class to be set
    return result;
  }

  protected Instances determineOutputFormat(Instances inputFormat) {
      return inputFormat;
      //Instances result = new Instances(inputFormat, 0);
      //result.insertAttributeAt(new Attribute("bla"), result.numAttributes());
      //return result;
  }

  protected Instance process(Instance instance) throws Exception {
   try {
    return (Instance) processFn.invoke((Object) instance);
   } catch (Exception e) { 
     // Weka silently eats any exceptions in filters so we need to report on it...
     System.out.println("Unable to filter an instance with a clojure fn!  The exception was:\n" +
                        e.getMessage());
     e.printStackTrace();
     throw e;
   } 
  }

  public static void main(String[] args) {
    //No-op
  }
}
