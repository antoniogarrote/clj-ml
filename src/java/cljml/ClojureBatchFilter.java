package cljml;

/**
 * A SimpleBatchFilter whose logic is composed of clojure functions.
 *
 * @author Ben Mabey
 */

import clojure.lang.IFn;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.SimpleBatchFilter;
import weka.core.Capabilities.*;
import weka.core.Capabilities;

public class ClojureBatchFilter
  extends SimpleBatchFilter {

  public IFn processFn;
  public IFn determineOutputFormatFn;

  public ClojureBatchFilter(IFn processFn) {
    this.processFn = processFn;
  }

  public ClojureBatchFilter(IFn processFn, IFn determineOutputFormatFn) {
    this.processFn = processFn;
    this.determineOutputFormatFn = determineOutputFormatFn;
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

  protected Instances determineOutputFormat(Instances inputFormat)  throws Exception {
    if (this.determineOutputFormatFn == null) {
        return inputFormat;
    } else {
      return (Instances) determineOutputFormatFn.invoke((Object) inputFormat);
    }
  }

  protected Instances process(Instances instances) throws Exception {
   try {
    return (Instances) processFn.invoke((Object) instances);
   } catch (Exception e) { 
     // Weka silently eats any exceptions in filters so we need to report on it...
     System.out.println("Unable to filter instances (dataset) with a clojure fn!  The exception was:\n" +
                        e.getMessage());
     e.printStackTrace();
     throw e;
   } 
  }

  public static void main(String[] args) {
    //No-op
  }
}
