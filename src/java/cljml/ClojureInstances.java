package cljml;

import clojure.lang.ISeq;
import java.io.IOException;
import java.util.Iterator;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
/**
 * A wrapper around Weka's Instances class to add some Clojure behavior.
 *
 * @author Antonio Garrote
 */

class ClojureInstancesIterator implements Iterator<Instance> {
    private Instances instances;
    private int counter;

    public ClojureInstancesIterator(Instances insts) {
        this.instances = insts;
        this.counter = 0;
    }
    public boolean hasNext() {
        return counter < instances.numInstances();
    }

    public Instance next() {
        Instance next = instances.instance(counter);
        counter++;
        return next;
    }

    public void remove() {
        instances.delete(counter - 1);
    }


}
public class ClojureInstances extends weka.core.Instances implements Iterable<weka.core.Instance>{
    public ClojureInstances(Instances dataset) {
        super(dataset);
    }

    public ClojureInstances(Instances dataset, int capacity) {
        super(dataset,capacity);
    }

    public ClojureInstances(Instances source, int first, int toCopy) {
        super(source, first, toCopy);
    }

    public ClojureInstances(java.io.Reader reader) throws IOException {
        super(reader);
    }

    public ClojureInstances(java.lang.String name, FastVector attInfo, int capacity) {
        super(name, attInfo, capacity);
    }

    public Iterator<Instance> iterator() {
        return new ClojureInstancesIterator(this);
    }

}
