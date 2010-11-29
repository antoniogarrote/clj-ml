;;
;; Classifiers
;; @author Antonio Garrote
;;

(ns #^{:author "Antonio Garrote <antoniogarrote@gmail.com>"}
  clj-ml.classifiers
  "This namespace contains several functions for building classifiers using different
   classification algorithms: Bayes networks, multilayer perceptron, decission tree or
   support vector machines are available. Some of these classifiers have incremental
   versions so they can be built without having all the dataset instances in memory.

   Functions for evaluating the classifiers built using cross validation or a training
   set are also provided.

   A sample use of the API for classifiers is shown below:

    (use 'clj-ml.classifiers)

    ; Building a classifier using a  C4.5 decission tree
    (def *classifier* (make-classifier :decission-tree :c45))

    ; We set the class attribute for the loaded dataset.
    ; *dataset* is supposed to contain a set of instances.
    (dataset-set-class *dataset* 4)

    ; Training the classifier
    (classifier-train *classifier* *dataset*)

    ; We evaluate the classifier using a test dataset
    (def *evaluation*   (classifier-evaluate *classifier* :dataset *dataset* *trainingset*))

    ; We retrieve some data from the evaluation result
    (:kappa *evaluation*)
    (:root-mean-squared-error *evaluation*)
    (:precision *evaluation*)

    ; A trained classifier can be used to classify new instances
    (def *to-classify* (make-instance *dataset*  {:class :Iris-versicolor
                                                  :petalwidth 0.2
                                                  :petallength 1.4
                                                  :sepalwidth 3.5
                                                  :sepallength 5.1}))

    ; We retrieve the index of the class value assigned by the classifier
    (classifier-classify *classifier* *to-classify*)

    ; We retrieve a symbol with the value assigned by the classifier
    ; and assigns it to a certain instance
    (classifier-label *classifier* *to-classify*)

   A classifier can also be trained using cross-validation:

    (classifier-evaluate *classifier* :cross-validation *dataset* 10)

   Finally a classifier can be stored in a file for later use:

    (use 'clj-ml.utils)

    (serialize-to-file *classifier*
     \"/Users/antonio.garrote/Desktop/classifier.bin\")
"
  (:use [clj-ml utils data kernel-functions])
  (:import (java.util Date Random)
           (weka.core Instance Instances)
           (weka.classifiers.trees J48)
           (weka.classifiers.bayes NaiveBayes NaiveBayesUpdateable)
           (weka.classifiers.functions MultilayerPerceptron SMO)
           (weka.classifiers Classifier Evaluation)))


;; Setting up classifier options

(defmulti #^{:skip-wiki true}
  make-classifier-options
  "Creates the right parameters for a classifier. Returns the parameters as a Clojure vector."
  (fn [kind algorithm map] [kind algorithm]))

(defmethod make-classifier-options [:decission-tree :c45]
  ([kind algorithm m]
     (->> (check-options m
                         {:unpruned "-U"
                          :reduced-error-pruning "-R"
                          :only-binary-splits "-B"
                          :no-raising "-S"
                          :no-cleanup "-L"
                          :laplace-smoothing "-A"})
          (check-option-values m
                               {:pruning-confidence "-C"
                                :minimum-instances "-M"
                                :pruning-number-folds "-N"
                                :random-seed "-Q"}))))

(defmethod make-classifier-options [:bayes :naive]
  ([kind algorithm m]
     (check-options m
                    {:kernel-estimator "-K"
                     :supervised-discretization "-D"
                     :old-format "-O"})))

(defmethod make-classifier-options [:neural-network :multilayer-perceptron]
  ([kind algorithm m]
     (->> (check-options m
                         {:no-nominal-to-binary "-B"
                          :no-numeric-normalization "-C"
                          :no-normalization "-I"
                          :no-reset "-R"
                          :learning-rate-decay "-D"})
          (check-option-values m
                               {:learning-rate "-L"
                                :momentum "-M"
                                :epochs "-N"
                                :percentage-validation-set "-V"
                                :random-seed "-S"
                                :threshold-number-errors "-E"}))))

(defmethod make-classifier-options [:support-vector-machine :smo]
  ([kind algorithm m]
     (->> (check-options m {:fit-logistic-models "-M"})
          (check-option-values m
                               {:complexity-constant "-C"
                                :tolerance "-L"
                                :epsilon-roundoff "-P"
                                :folds-for-cross-validation "-V"
                                :random-seed "-W"}))))


;; Building classifiers


(defn make-classifier-with
  #^{:skip-wiki true}
  [kind algorithm ^Class classifier-class options]
     (let [options-read (if (empty? options)  {} (first options))
            ^Classifier classifier (.newInstance classifier-class)
           opts (into-array String (make-classifier-options kind algorithm options-read))]
       (.setOptions classifier opts)
        classifier))

(defmulti make-classifier
  "Creates a new classifier for the given kind algorithm and options.

   The first argument identifies the kind of classifier and the second
   argument the algorithm to use, e.g. :decission-tree :c45.

   The classifiers currently supported are:

     - :decission-tree :c45
     - :bayes :naive
     - :neural-network :mutilayer-perceptron
     - :support-vector-machine :smo

   Optionally, a map of options can also be passed as an argument with
   a set of classifier specific options.

   This is the description of the supported classifiers and the accepted
   option parameters for each of them:

    * :decission-tree :c45

      A classifier building a pruned or unpruned C 4.5 decission tree using
      Weka J 4.8 implementation.

      Parameters:

        - :unpruned
            Use unpruned tree. Sample value: true
        - :reduce-error-pruning
            Sample value: true
        - :only-binary-splits
            Sample value: true
        - :no-raising
            Sample value: true
        - :no-cleanup
            Sample value: true
        - :laplace-smoothing
            For predicted probabilities. Sample value: true
        - :pruning-confidence
            Threshold for pruning. Default value: 0.25
        - :minimum-instances
            Minimum number of instances per leave. Default value: 2
        - :pruning-number-folds
            Set number of folds for reduced error pruning. Default value: 3
        - :random-seed
            Seed for random data shuffling. Default value: 1

    * :bayes :naive

      Classifier based on the Bayes' theorem with strong independence assumptions, among the
      probabilistic variables.

      Parameters:

        - :kernel-estimator
            Use kernel desity estimator rather than normal. Sample value: true
        - :supervised-discretization
            Use supervised discretization to to process numeric attributes (see :supervised-discretize
            filter in clj-ml.filters/make-filter function). Sample value: true

    * :neural-network :multilayer-perceptron

      Classifier built using a feedforward artificial neural network with three or more layers
      of neurons and nonlinear activation functions. It is able to distinguish data that is not
      linearly separable.

      Parameters:

        - :no-nominal-to-binary
            A :nominal-to-binary filter will not be applied by default. (see :supervised-nominal-to-binary
            filter in clj-ml.filters/make-filter function). Default value: false
        - :no-numeric-normalization
            A numeric class will not be normalized. Default value: false
        - :no-nomalization
            No attribute will be normalized. Default value: false
        - :no-reset
            Reseting the network will not be allowed. Default value: false
        - :learning-rate-decay
            Learning rate decay will occur. Default value: false
        - :learning-rate
            Learning rate for the backpropagation algorithm. Value should be between [0,1].
            Default value: 0.3
        - :momentum
            Momentum rate for the backpropagation algorithm. Value shuld be between [0,1].
            Default value: 0.2
        - :epochs
            Number of iteration to train through. Default value: 500
        - :percentage-validation-set
            Percentage size of validation set to use to terminate training. If it is not zero
            it takes precende over the number of epochs to finish training. Values should be
            between [0,100]. Default value: 0
        - :random-seed
            Value of the seed for the random generator. Values should be longs greater than
            0. Default value: 1
        - :threshold-number-errors
            The consequetive number of errors allowed for validation testing before the network
            terminates. Values should be greater thant 0. Default value: 20

    * :support-vector-machine :smo

      Support vector machine (SVM) classifier built using the sequential minimal optimization (SMO)
      training algorithm.

      Parameters:

        - :fit-logistic-models
            Fit logistic models to SVM outputs. Default value :false
        - :complexity-constant
            The complexity constance. Default value: 1
        - :tolerance
            Tolerance parameter. Default value: 1.0e-3
        - :epsilon-roundoff
            Epsilon round-off error. Default value: 1.0e-12
        - :folds-for-cross-validation
            Number of folds for the internal cross-validation. Sample value: 10
        - :random-seed
            Value of the seed for the random generator. Values should be longs greater than
            0. Default value: 1
"
  (fn [kind algorithm & options] [kind algorithm]))

(defmethod make-classifier [:decission-tree :c45]
  ([kind algorithm & options]
     (make-classifier-with kind algorithm J48 options)))

(defmethod make-classifier [:bayes :naive]
  ([kind algorithm & options]
     (if (or (nil? (:updateable (first options)))
             (= (:updateable (first options)) false))
       (make-classifier-with kind algorithm NaiveBayes options)
       (make-classifier-with kind algorithm NaiveBayesUpdateable options))))

(defmethod make-classifier [:neural-network :multilayer-perceptron]
  ([kind algorithm & options]
     (make-classifier-with kind algorithm MultilayerPerceptron options)))

(defmethod make-classifier [:support-vector-machine :smo]
  ([kind algorithm & options]
     (let [options-read (if (empty? options)  {} (first options))
           classifier (new SMO)
           opts (into-array String (make-classifier-options :support-vector-machine :smo options-read))]
       (.setOptions classifier opts)
       (when (not (empty? (get options-read :kernel-function)))
          ;; We have to setup a different kernel function
         (let [kernel (get options-read :kernel-function)
                real-kernel (if (map? kernel)
                             (make-kernel-function (first (keys kernel))
                                                   (first (vals kernel)))
                             kernel)]
            (.setKernel classifier real-kernel)))
        classifier)))

;; Training classifiers

(defn classifier-train
  "Trains a classifier with the given dataset as the training data."
  ([^Classifier classifier dataset]
     (do (.buildClassifier classifier dataset)
         classifier)))

(defn classifier-update
  "If the classifier is updateable it updates the classifier with the given instance or set of instances."
  ([^Classifier classifier instance-s]
     ;; Arg... weka doesn't provide a formal interface for updateClassifer- How do I type hint this?
     (if (is-dataset? instance-s)
       (do (doseq [i (dataset-seq instance-s)]
             (.updateClassifier classifier ^Instance i))
           classifier)
       (do (.updateClassifier classifier ^Instance instance-s)
           classifier))))

;; Evaluating classifiers

(defn- collect-evaluation-results
   "Collects all the statistics from the evaluation of a classifier."
  ([class-values ^Evaluation evaluation]
     (do
       (println (.toMatrixString evaluation))
       (println "=== Summary ===")
       (println (.toSummaryString evaluation))
       {:correct (try-metric #(.correct evaluation))
        :incorrect (try-metric #(.incorrect evaluation))
        :unclassified (try-metric #(.unclassified evaluation))
        :percentage-correct (try-metric #(.pctCorrect evaluation))
        :percentage-incorrect (try-metric #(.pctIncorrect evaluation))
        :percentage-unclassified (try-metric #(.pctUnclassified evaluation))
        :error-rate (try-metric #(.errorRate evaluation))
        :mean-absolute-error (try-metric #(.meanAbsoluteError evaluation))
        :relative-absolute-error (try-metric #(.relativeAbsoluteError evaluation))
        :root-mean-squared-error (try-metric #(.rootMeanSquaredError evaluation))
        :root-relative-squared-error (try-metric #(.rootRelativeSquaredError evaluation))
        :correlation-coefficient (try-metric #(.correlationCoefficient evaluation))
        :average-cost (try-metric #(.avgCost evaluation))
        :kappa (try-metric #(.kappa evaluation))
        :kb-information (try-metric #(.KBInformation evaluation))
        :kb-mean-information (try-metric #(.KBMeanInformation evaluation))
        :kb-relative-information (try-metric #(.KBRelativeInformation evaluation))
        :sf-entropy-gain (try-metric #(.SFEntropyGain evaluation))
        :sf-mean-entropy-gain (try-metric #(.SFMeanEntropyGain evaluation))
        :roc-area (try-multiple-values-metric class-values (fn [i] (try-metric #(.areaUnderROC evaluation i))))
        :false-positive-rate (try-multiple-values-metric class-values (fn [i] (try-metric #(.falsePositiveRate evaluation i))))
        :false-negative-rate (try-multiple-values-metric class-values (fn [i] (try-metric #(.falseNegativeRate evaluation i))))
        :f-measure (try-multiple-values-metric class-values (fn [i] (try-metric #(.fMeasure evaluation i))))
        :precision (try-multiple-values-metric class-values (fn [i] (try-metric #(.precision evaluation i))))
        :recall (try-multiple-values-metric class-values (fn [i] (try-metric #(.recall evaluation i))))
        :evaluation-object evaluation})))

(defmulti classifier-evaluate
  "Evaluates a trained classifier using the provided dataset or cross-validation.
   The first argument must be the classifier to evaluate, the second argument is
   the kind of evaluation to do.
   Two possible evaluations ara availabe: dataset and cross-validations. The values
   for the second argument can be:

    - :dataset
    - :cross-validation

    * :dataset

    If dataset evaluation is desired, the function call must receive as the second
    parameter the keyword :dataset and as third and fourth parameters the original
    dataset used to build the classifier and the training data:

      (classifier-evaluate *classifier* :dataset *training* *evaluation*)

    * :cross-validation

    If cross-validation is desired, the function call must receive as the second
    parameter the keyword :cross-validation and as third and fourth parameters the dataset
    where for training and the number of folds.

      (classifier-evaluate *classifier* :cross-validation *training* 10)

    The metrics available in the evaluation are listed below:

    - :correct
        Number of instances correctly classified
    - :incorrect
        Number of instances incorrectly evaluated
    - :unclassified
        Number of instances incorrectly classified
    - :percentage-correct
        Percentage of correctly classified instances
    - :percentage-incorrect
        Percentage of incorrectly classified instances
    - :percentage-unclassified
        Percentage of not classified instances
    - :error-rate
    - :mean-absolute-error
    - :relative-absolute-error
    - :root-mean-squared-error
    - :root-relative-squared-error
    - :correlation-coefficient
    - :average-cost
    - :kappa
        The kappa statistic
    - :kb-information
    - :kb-mean-information
    - :kb-relative-information
    - :sf-entropy-gain
    - :sf-mean-entropy-gain
    - :roc-area
    - :false-positive-rate
    - :false-negative-rate
    - :f-measure
    - :precision
    - :recall
    - :evaluation-object
        The underlying Weka's Java object containing the evaluation
"
  (fn [classifier mode & evaluation-data] mode))

(defmethod classifier-evaluate :dataset
  ([^Classifier classifier mode & [training-data test-data]]
     (let [evaluation (new Evaluation training-data)
           class-values (dataset-class-values training-data)]
       (.evaluateModel evaluation classifier test-data (into-array []))
       (collect-evaluation-results class-values evaluation))))

(defmethod classifier-evaluate :cross-validation
  ([classifier mode & [training-data folds]]
     (let [evaluation (new Evaluation training-data)
           class-values (dataset-class-values training-data)]
       (.crossValidateModel evaluation classifier training-data folds (new Random (.getTime (new Date))) (into-array []))
       (collect-evaluation-results class-values evaluation))))


;; Classifying instances

(defn classifier-classify
  "Classifies an instance using the provided classifier.
   The value returned is the numeric attribute of that value for
   the list of valid values for the class."
  ([^Classifier classifier ^Instance instance]
     (.classifyInstance classifier instance)))

(defn classifier-label
  "Classifies and assign a label to a dataset instance.
   This function is similar to classifier-classify but
   instead of just returning the numeric identifier for the
   new instance, it changes the class value for that instance
   to the newly assigned by the classifier.

   The function returns the newly classified instance.

   This call is destructive, the instance passed as an argument
   is modified.

    ; We create the instance to classify
    (def *to-classify* (make-instance *dataset*  {:class :Iris-versicolor
                                                  :petalwidth 0.2
                                                  :petallength 1.4
                                                  :sepalwidth 3.5
                                                  :sepallength 5.1}))

    ; We use the classifier to check the value for the class
    (classifier-classify *classifier* *to-classify*)
     >0.0

    ; We change the class for the instance according to the assigned class
    (classifier-label *classifier* *to-classify*)
     >#<Instance 5.1,3.5,1.4,0.2,Iris-setosa>
"
  ([classifier instance]
     (let [cls (classifier-classify classifier instance)]
       (instance-set-class instance cls))))
