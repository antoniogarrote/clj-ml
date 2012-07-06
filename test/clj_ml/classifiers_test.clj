(ns clj-ml.classifiers-test
  (:use [clj-ml classifiers data] :reload-all)
  (:use clojure.test midje.sweet))


(deftest make-classifiers-options-c45
  (fact
    (let [options (make-classifier-options :decision-tree :c45 {:unpruned true :reduced-error-pruning true :only-binary-splits true :no-raising true
                                                                :no-cleanup true :laplace-smoothing true :pruning-confidence 0.12 :minimum-instances 10
                                                                :pruning-number-folds 5 :random-seed 1})]
      options => (just ["-U" "-R" "-B" "-S" "-L" "-A" "-C" "0.12" "-M" "10" "-N" "5" "-Q" "1"] :in-any-order))))


(deftest make-classifier-c45
  (let [c (make-classifier :decision-tree :c45)]
    (is (= (class c)
           weka.classifiers.trees.J48))))

(deftest train-classifier-c45
  (let [c (make-classifier :decision-tree :c45)
        ds (clj-ml.data/make-dataset "test" [:a :b {:c [:m :n]}] [[1 2 :m] [4 5 :m]])]
    (clj-ml.data/dataset-set-class ds 2)
    (classifier-train c ds)
    (is true)))

(deftest make-classifier-bayes
  (fact
    (let [c (clj-ml.classifiers/make-classifier :bayes :naive {:kernel-estimator true :old-format true})
          opts (vec (.getOptions c))]
      opts => (contains ["-K" "-O"]))))

(deftest make-classifier-bayes-updateable
  (let [c (clj-ml.classifiers/make-classifier :bayes :naive {:updateable true})]
    (is (= (class c)
           weka.classifiers.bayes.NaiveBayesUpdateable))))

(deftest train-classifier-bayes
  (let [c (clj-ml.classifiers/make-classifier :bayes :naive {:kernel-estimator true :old-format true})
        ds (clj-ml.data/make-dataset "test" [:a :b {:c [:m :n]}] [[1 2 :m] [4 5 :m]])]
    (clj-ml.data/dataset-set-class ds 2)
    (classifier-train c ds)
    (is true)))


(deftest classifier-evaluate-dataset
  (let [c   (make-classifier :decision-tree :c45)
        ds  (clj-ml.data/make-dataset "test" [:a :b {:c [:m :n]}] [[1 2 :m] [4 5 :m]])
        tds (clj-ml.data/make-dataset "test" [:a :b {:c [:m :n]}] [[4 1 :n] [4 5 :m]])
        _   (clj-ml.data/dataset-set-class ds 2)
        _   (clj-ml.data/dataset-set-class tds 2)
        _   (classifier-train c ds)
        res (classifier-evaluate c :dataset ds tds)]
    (is (= 26 (count (keys res))))))

(deftest make-classifier-svm-smo-polykernel
  (let [svm (make-classifier :support-vector-machine :smo {:kernel-function {:polynomic {:exponent 2.0}}})]
    (is (= weka.classifiers.functions.supportVector.PolyKernel
           (class (.getKernel svm))))))


(deftest classifier-evaluate-cross-validation
  (let [c (make-classifier :decision-tree :c45)
        ds (clj-ml.data/make-dataset "test" [:a :b {:c [:m :n]}] [[1 2 :m] [4 5 :m]])
        _  (clj-ml.data/dataset-set-class ds 2)
        _  (classifier-train c ds)
        res (classifier-evaluate c :cross-validation ds 2)]
    (is (= 26 (count (keys res))))))

(deftest update-updateable-classifier
  (let [c (clj-ml.classifiers/make-classifier :bayes :naive {:updateable true})
        ds (clj-ml.data/make-dataset "test" [:a :b {:c [:m :n]}] [[1 2 :m] [4 5 :m]])
        _  (dataset-set-class ds 2)
        inst (make-instance ds {:a 56 :b 45 :c :m})]
    (classifier-train  c ds)
    (classifier-update c ds)
    (classifier-update c inst)
    (is true)))
