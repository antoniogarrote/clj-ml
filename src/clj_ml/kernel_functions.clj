;;
;; Kernel functions for SVM classifiers
;; @author Antonio Garrote
;;

(ns clj-ml.kernel-functions
  (:use [clj-ml utils data])
  (:import (weka.classifiers.functions.supportVector PolyKernel RBFKernel StringKernel)))

(defmulti make-kernel-function-options
  "Creates ther right parameters for a kernel-function"
  (fn [kind map] kind))

(defmethod make-kernel-function-options :polynomic
  ([kind map]
     (let [cols-val (check-option-values {:cache-size "-C"
                                          :exponent "-E"
                                          :use=lower-order-terms "-L"}
                                         map
                                         [""])]
       (into-array cols-val))))


(defmethod make-kernel-function-options :radial-basis
  ([kind map]
     (let [cols-val (check-option-values {:cache-size "-C"
                                          :gamma "-G"}
                                         map
                                         [""])]
       (into-array cols-val))))

(defmethod make-kernel-function-options :string
  ([kind map]
     (let [pre-values-a (if (get map :use-normalization)
                          (if (get map :use-normalization)
                            ["-N" "yes"]
                            ["-N" "no"])
                          [""])
           _foo (println (str "pre a" pre-values-a " map " map))
           pre-values-b (if (get map :pruning)
                          (if (= (get map :pruning)
                                 :lambda)
                            (conj (conj pre-values-a "-P") "1")
                            (conj (conj pre-values-a "-P" ) "0"))
                          pre-values-a)
           _foo (println (str "pre b" pre-values-b))
           cols-val (check-option-values {:cache-size "-C"
                                          :internal-cache-size "-IC"
                                          :lambda "-L"
                                          :sequence-length "-ssl"
                                          :maximum-sequence-length "-ssl-max"}
                                         map
                                         pre-values-b)]
       (into-array cols-val))))

(defmulti make-kernel-function
  "Creates a new kernel function"
  (fn [kind & options] kind))


(defmethod make-kernel-function :polynomic
  ([kind & options]
     (let [dist (new PolyKernel)
           opts (make-kernel-function-options :polynomic (first-or-default options {}))]
       (.setOptions dist opts)
       dist)))

(defmethod make-kernel-function :radial-basis
  ([kind & options]
     (let [dist (new RBFKernel)
           opts (make-kernel-function-options :radial-basis (first-or-default options {}))]
       (.setOptions dist opts)
       dist)))

(defmethod make-kernel-function :string
  ([kind & options]
     (let [dist (new StringKernel)
           opts (make-kernel-function-options :string (first-or-default options {}))]
       (.setOptions dist opts)
       dist)))

