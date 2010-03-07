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
   set are also provided"
  (:use [clj-ml utils data kernel-functions])
  (:import (java.util Date Random)
           (weka.classifiers.trees J48)
           (weka.classifiers.bayes NaiveBayes)
           (weka.classifiers.bayes NaiveBayesUpdateable)
           (weka.classifiers.functions MultilayerPerceptron)
           (weka.classifiers.functions SMO)
           (weka.classifiers Evaluation)))


;; Setting up classifier options

(defmulti #^{:skip-wiki true}
  make-classifier-options
  "Creates the right parameters for a classifier"
  (fn [kind algorithm map] [kind algorithm]))

(defmethod make-classifier-options [:decission-tree :c45]
  ([kind algorithm map]
     (let [cols-val (check-options {:unpruned "-U"
                                    :reduced-error-pruning "-R"
                                    :only-binary-splits "-B"
                                    :no-raising "-S"
                                    :no-cleanup "-L"
                                    :laplace-smoothing "-A"}
                                     map
                                     [""])
           cols-val-a (check-option-values {:pruning-confidence "-C"
                                            :minimum-instances "-M"
                                            :pruning-number-folds "-N"
                                            :shuffling-random-seed "-Q"}
                                           map
                                           cols-val)]
    (into-array cols-val-a))))

(defmethod make-classifier-options [:bayes :naive]
  ([kind algorithm map]
     (let [cols-val (check-options {:kernel-estimator "-K"
                                    :supervised-discretization "-D"
                                    :old-format "-O"}
                                     map
                                     [""])]
       (into-array cols-val))))

(defmethod make-classifier-options [:neural-network :multilayer-perceptron]
  ([kind algorithm map]
     (let [cols-val (check-options {:no-nominal-to-binary "-B"
                                    :no-numeric-normalization "-C"
                                    :no-normalization "-I"
                                    :no-reset "-R"
                                    :learning-rate-decay "-D"}
                                     map
                                     [""])
           cols-val-a (check-option-values {:learning-rate "-L"
                                            :momentum "-M"
                                            :epochs "-N"
                                            :percentage-validation-set "-V"
                                            :seed "-S"
                                            :threshold-number-errors "-E"}
                                           map
                                           cols-val)]
       (into-array cols-val-a))))

(defmethod make-classifier-options [:support-vector-machine :smo]
  ([kind algorithm map]
     (let [cols-val (check-options {:fit-logistic-models "-M"}
                                     map
                                     [""])
           cols-val-a (check-option-values {:complexity-constant "-C"
                                            :tolerance "-L"
                                            :epsilon-roundoff "-P"
                                            :folds-for-cross-validation "-V"
                                            :random-seed "-W"}
                                           map
                                           cols-val)]
       (into-array cols-val-a))))


;; Building classifiers


(defmacro make-classifier-m
  #^{:skip-wiki true}
  ([kind algorithm classifier-class options]
     `(let [options-read# (if (empty? ~options)  {} (first ~options))
            classifier# (new ~classifier-class)
            opts# (make-classifier-options ~kind ~algorithm options-read#)]
        (.setOptions classifier# opts#)
        classifier#)))

(defmulti make-classifier
  "Creates a new classifier for the given kind algorithm and options.

   The first argument identifies the kind of classifier and the second
   argument the algorithm to use, e.g. :decission-tree :c45.

   The colection of classifiers currently supported are:

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

       - :unpruned Use unpruned tree. Sample value: true
       - :reduce-error-pruning Sample value: true
       - :only-binary-splits Sample value: true
       - :no-raising Sample value: true
       - :no-cleanup Sample value: true
       - :laplace-smoothing For predicted probabilities. Sample value: true
       - :pruning-confidence Threshold for pruning. Default value: 0.25
       - :minimum-instances Minimum number of instances per leave. Default
                            value: 2
       - :pruning-number-folds Set number of folds for reduced error pruning.
                               Default value: 3
       - :shuffling-random-seed Seed for random data shuffling. Default value: 1
    "
  (fn [kind algorithm & options] [kind algorithm]))

(defmethod make-classifier [:decission-tree :c45]
  ([kind algorithm & options]
     (make-classifier-m kind algorithm J48 options)))

(defmethod make-classifier [:bayes :naive]
  ([kind algorithm & options]
     (if (or (nil? (:updateable (first options)))
             (= (:updateable (first options)) false))
       (make-classifier-m kind algorithm NaiveBayes options)
       (make-classifier-m kind algorithm NaiveBayesUpdateable options))))

(defmethod make-classifier [:neural-network :multilayer-perceptron]
  ([kind algorithm & options]
     (make-classifier-m kind algorithm MultilayerPerceptron options)))

(defmethod make-classifier [:support-vector-machine :smo]
  ([kind algorithm & options]
     (let [options-read (if (empty? options)  {} (first options))
           classifier (new SMO)
           opts (make-classifier-options :support-vector-machine :smo options-read)]
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
  "Trains a classifier with the given dataset as the training data"
  ([classifier dataset]
     (do (.buildClassifier classifier dataset)
         classifier)))

(defn classifier-update
  "If the classifier is updateable it updates the classifier with the given instance or set of instances"
  ([classifier instance-s]
     (if (is-dataset? instance-s)
       (do (for [i (dataset-seq instance-s)]
             (.updateClassifier classifier i))
           classifier)
       (do (.updateClassifier classifier instance-s)
           classifier))))

;; Evaluating classifiers

(defn- collect-evaluation-results
  "Collects all the statistics from the evaluation of a classifier"
  ([class-values evaluation]
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
  "Evaluetes a trained classifier using the provided dataset or cross-validation"
  (fn [classifier mode & evaluation-data] mode))

(defmethod classifier-evaluate :dataset
  ([classifier mode & evaluation-data]
     (let [training-data (nth evaluation-data 0)
           test-data (nth evaluation-data 1)
           evaluation (new Evaluation training-data)
           class-values (dataset-class-values training-data)]
       (.evaluateModel evaluation classifier test-data (into-array []))
       (collect-evaluation-results class-values evaluation))))

(defmethod classifier-evaluate :cross-validation
  ([classifier mode & evaluation-data]
     (let [training-data (nth evaluation-data 0)
           folds (nth evaluation-data 1)
           evaluation (new Evaluation training-data)
           class-values (dataset-class-values training-data)]
       (.crossValidateModel evaluation classifier training-data folds (new Random (.getTime (new Date))) (into-array []))
       (collect-evaluation-results class-values evaluation))))


;; Classifying instances

(defn classifier-classify
  "Classifies an instance or data vector using the provided classifier"
  ([classifier instance]
     (.classifyInstance classifier instance)))

(defn classifier-label
  "Classifies and assign a label to a dataset instance"
  ([classifier instance]
     (let [cls (classifier-classify classifier instance)]
       (instance-set-class instance cls))))
