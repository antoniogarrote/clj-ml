# clj-ml

A machine learning library for Clojure built on top of Weka and friends

## Installation

In order to install the library you must first install Leiningen.

### To install from source

git clone the project, then run:

    $ lein deps
    $ lein javac
    $ lein uberjar

### Installing from Clojars

  [clj-ml "0.0.3-SNAPSHOT"]

### Installing from Maven

(add Clojars repository)

    <dependency>
      <groupId>clj-ml</groupId>
      <artifactId>clj-ml</artifactId>
      <version>0.0.3-SNAPSHOT</version>
    </dependency>

## Supported algorithms

 * Filters
   * supervised discretize
   * unsupervised discretize
   * supervised nominal to binary
   * unsupervised nominal to binary

 * Classifiers
   * C4.5 (J4.8)
   * naive Bayes
   * multilayer perceptron

 * Clusterers
   * k-means

## Usage

### I/O of data

    REPL>(use 'clj-ml.io)

    REPL>; Loading data from an ARFF file, XRFF and CSV are also supported
    REPL>(def ds (load-instances :arff "file:///Applications/weka-3-6-2/data/iris.arff"))

    REPL>; Saving data in a different format
    REPL>(save-instances :csv "file:///Users/antonio.garrote/Desktop/iris.csv"  ds)

### Working with datasets

    REPL>(use 'clj-ml.data)

    REPL>; Defining a dataset
    REPL>(def ds (make-dataset "name" [:length :width {:kind [:good :bad]}] [ [12 34 :good] [24 53 :bad] ]))
    REPL>ds

    #<ClojureInstances @relation name

    @attribute length numeric
    @attribute width numeric
    @attribute kind {good,bad}

    @data
    12,34,good
    24,53,bad>

    REPL>; Using datasets like sequences
    REPL>(dataset-seq ds)

    (#<Instance 12,34,good> #<Instance 24,53,bad>)

    REPL>; Transforming instances  into maps or vectors
    REPL>(instance-to-map (first (dataset-seq ds)))

    {:kind :good, :width 34.0, :length 12.0}

    REPL>(instance-to-vector (dataset-at ds 0))
    [12.0 34.0 :good]

### Filtering datasets

    REPL>(use '(clj-ml filters io))

    REPL>(def ds (load-instances :arff "file:///Applications/weka-3-6-2/data/iris.arff"))

    REPL>; Discretizing a numeric attribute using an unsupervised filter
    REPL>(def  discretize (make-filter :unsupervised-discretize {:dataset-format ds :attributes [0 2]}))


    REPL>(def filtered-ds (filter-apply discretize ds))

    REPL>; The eqivalent operation can be done with the ->> macro and make-apply-filter fn:
    REPL>(def filtered-ds (->> "file:///Applications/weka-3-6-2/data/iris.arff")
                               (load-instances :arff) 
                               (make-apply-filter :unsupervised-discretize {:attributes [0 2]}))

### Using classifiers

    REPL>(use 'clj-ml.classifiers)

    REPL>; Building a classifier using a  C4.5 decission tree
    REPL>(def classifier (make-classifier :decission-tree :c45))

    REPL>; We set the class attribute for the loaded dataset
    REPL>(dataset-set-class ds 4)

    REPL>; Training the classifier
    REPL>(classifier-train classifier ds)

     #<J48 J48 pruned tree
     ------------------

     petalwidth <= 0.6: Iris-setosa (50.0)
     petalwidth > 0.6
     |	petalwidth <= 1.7
     |	|   petallength <= 4.9: Iris-versicolor (48.0/1.0)
     |	|   petallength > 4.9
     |	|   |	petalwidth <= 1.5: Iris-virginica (3.0)
     |	|   |	petalwidth > 1.5: Iris-versicolor (3.0/1.0)
     |	petalwidth > 1.7: Iris-virginica (46.0/1.0)

     Number of Leaves  :		5

     Size of the tree :	9


    REPL>; We evaluate the classifier using a test dataset
    REPL>; last parameter should be a different test dataset, here we are using the same
    REPL>(def evaluation   (classifier-evaluate classifier  :dataset ds ds))

     === Confusion Matrix ===

       a	 b  c	<-- classified as
      50	 0  0 |	 a = Iris-setosa
       0 49  1 |	 b = Iris-versicolor
       0	 2 48 |	 c = Iris-virginica

     === Summary ===

     Correctly Classified Instances	   147		     98	     %
     Incorrectly Classified Instances	     3		      2	     %
     Kappa statistic			     0.97
     Mean absolute error			     0.0233
     Root mean squared error		     0.108
     Relative absolute error		     5.2482 %
     Root relative squared error		    22.9089 %
     Total Number of Instances		   150

    REPL>(:kappa evaluation)

     0.97

    REPL>(:root-mean-squared-error e)

     0.10799370769526968

    REPL>(:precision e)

     {:Iris-setosa 1.0, :Iris-versicolor 0.9607843137254902, :Iris-virginica
      0.9795918367346939}

    REPL>; The classifier can also be evaluated using cross-validation
    REPL>(classifier-evaluate classifier :cross-validation ds 10)

     === Confusion Matrix ===

       a	 b  c	<-- classified as
      49	 1  0 |	 a = Iris-setosa
       0 47  3 |	 b = Iris-versicolor
       0	 4 46 |	 c = Iris-virginica

     === Summary ===

     Correctly Classified Instances	   142		     94.6667 %
     Incorrectly Classified Instances	     8		      5.3333 %
     Kappa statistic			     0.92
     Mean absolute error			     0.0452
     Root mean squared error		     0.1892
     Relative absolute error		    10.1707 %
     Root relative squared error		    40.1278 %
     Total Number of Instances		   150

    REPL>; A trained classifier can be used to classify new instances
    REPL>(def to-classify (make-instance ds
                                                      {:class :Iris-versicolor,
                                                      :petalwidth 0.2,
                                                      :petallength 1.4,
                                                      :sepalwidth 3.5,
                                                      :sepallength 5.1}))
    REPL>(classifier-classify classifier to-classify)

     0.0

    REPL>(classifier-label to-classify)

     #<Instance 5.1,3.5,1.4,0.2,Iris-setosa>
     ; FIXME: Outdated, it blows up with:
     ; java.lang.IllegalArgumentException: Wrong number of args passed to: classifiers$classifier-label (NO_SOURCE_FILE:0)


    REPL>; The classifiers can be saved and restored later
    REPL>(use 'clj-ml.utils)

    REPL>(serialize-to-file classifier "/Users/antonio.garrote/Desktop/classifier.bin")

### Using clusterers

    REPL>(use 'clj-ml.clusterers)

    REPL> ; we build a clusterer using k-means and three clusters
    REPL> (def kmeans (make-clusterer :k-means {:number-clusters 3}))

    REPL> ; we need to remove the class from the dataset to
    REPL> ; use this clustering algorithm
    REPL> (dataset-remove-class ds)
          ; FIXME: Outdated - there is no dataset-remove-class

    REPL> ; we build the clusters
    REPL> (clusterer-build kmeans ds)
    REPL> kmeans

      #<SimpleKMeans
      kMeans
      ======

      Number of iterations: 3
      Within cluster sum of squared errors: 7.817456892309574
      Missing values globally replaced with mean/mode

      Cluster centroids:
                                                Cluster#
      Attribute                Full Data               0               1               2
                                   (150)            (50)            (50)            (50)
      ==================================================================================
      sepallength                 5.8433           5.936           5.006           6.588
      sepalwidth                   3.054            2.77           3.418           2.974
      petallength                 3.7587            4.26           1.464           5.552
      petalwidth                  1.1987           1.326           0.244           2.026
      class                  Iris-setosa Iris-versicolor     Iris-setosa  Iris-virginica


## License

MIT License
