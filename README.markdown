# clj-ml

A machine learning library for Clojure built on top of Weka and friends

## Usage

* I/O of data

Loading data from a CSV file:

    (use 'clj-ml.io)

    ; Loading data from an ARFF file, XRFF and CSV are also supported
    (def ds (load-instances :arff "file:///Applications/weka-3-6-2/data/iris.arff"))

    ; Saving data in a different format
    (save-instances :csv ds)

* Working with datasets

    (use 'clj-ml.data)

    ; Defining a dataset
    (def ds (make-dataset ; name of the dataset
                    "name"
                    ; two numeric attributes and one nominal
                    [:length :width {:kind [:good :bad]}]
                    ; initial data
                     [12 34 :good]
                     [24 53 :bad] ]))

    ds

     #<ClojureInstances @relation name

     @attribute length numeric
     @attribute width numeric
     @attribute kind {good,bad}

     @data
     12,34,good
     24,53,bad>

    ; Using datasets like sequences
    (dataset-seq ds)

     (#<Instance 12,34,good> #<Instance 24,53,bad>)

    ; Transforming instances  into maps or vectors
    (instance-to-map (first (dataset-seq ds)))

     {:kind :good, :width 34.0, :length 12.0}

     (instance-to-vector (dataset-at ds 0))

* Filtering datasets

    (us 'clj-ml.filters)

    (def ds (load-instances :arff
    "file:///Applications/weka-3-6-2/data/iris.arff"))

    ; Discretizing a numeric attribute using an unsupervised filter
    (def  discretize (make-filter :unsupervised-discretize
                                                 {:dataset *ds*
                                                  :attributes [0 2]}))

    (def filtered-ds (filter-process discretize ds))

* Using classifiers

    (use 'clj-ml.classifiers)

    ; Building a classifier using a  C4.5 decission tree
    (def classifier (make-classifier :decission-tree :c45))

    ; We set the class attribute for the loaded dataset
    (dataset-set-class ds 4)

    ; Training the classifier
    (classifier-train classifier ds)

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


    ; We evaluate the classifier using a test dataset
    ; last parameter should be a different test dataset, here we are using the same
    (def evaluation   (classifier-evaluate classifier  :dataset ds ds))

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

    (:kappa evaluation)

     0.97

    (:root-mean-squared-error e)

     0.10799370769526968

    (:precision e)

     {:Iris-setosa 1.0, :Iris-versicolor 0.9607843137254902, :Iris-virginica
      0.9795918367346939}

    ; The classifier can also be evaluated using cross-validation
    (classifier-evaluate classifier :cross-validation ds 10)

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

    ; A trained classifier can be used to classify new instances
    (def to-classify (make-instance ds
                                                      {:class :Iris-versicolor,
                                                      :petalwidth 0.2,
                                                      :petallength 1.4,
                                                      :sepalwidth 3.5,
                                                      :sepallength 5.1}))
    (classifier-classify classifier to-classify)

     0.0

    (classifier-label to-classify)

     #<Instance 5.1,3.5,1.4,0.2,Iris-setosa>


    ; The classifiers can be saved and restored later
    (use 'clj-ml.utils)

    (serialize-to-file classifier
    "/Users/antonio.garrote/Desktop/classifier.bin")

## Installation

In order to install the library you must first install Leiningen.
You should also download the Weka 3.6.2 jar from the official weka homepage.
If maven complains about not finding weka, follow its instructions to install
the jar manually.

### To install from source

*  git clone the project
* $ lein deps
* $ lein compile
* $ lein compile-java
* $ lein uberjar

## License

MIT License
