;;
;; Data processing of data with different filtering algorithms
;; @author Antonio Garrote
;;

(ns clj-ml.filters
  (:use [clj-ml data utils])
  (:import (weka.filters Filter)))



;; Options for the filters

(defmulti make-filter-options
  "Creates the right parameters for a filter"
  (fn [kind map] kind))

(defmethod make-filter-options :supervised-discretize
  ([kind map]
     (let [cols (get map :attributes)
           pre-cols (reduce #(str %1 "," (+ %2 1)) "" cols)
           cols-val-a ["-R" (.substring pre-cols 1 (.length pre-cols))]
           cols-val-b (check-options {:invert "-V"
                                      :binary "-D"
                                      :better-encoding "-E"
                                      :kononenko "-K"}
                                     map
                                     cols-val-a)]
    (into-array cols-val-b))))

(defmethod make-filter-options :unsupervised-discretize
  ([kind map]
     (let [cols (get map :attributes)
           pre-cols (reduce #(str %1 "," (+ %2 1)) "" cols)
           cols-val-a ["-R" (.substring pre-cols 1 (.length pre-cols))]
           cols-val-b (check-options {:unset-class "-unset-class-temporarily"
                                      :binary "-D"
                                      :better-encoding "-E"
                                      :equal-frequency "-F"
                                      :optimize "-O"}
                                     map
                                     cols-val-a)
           cols-val-c (check-option-values {:number-bins "-B"
                                            :weight-bins "-M"}
                                           map
                                           cols-val-b)]
       (into-array cols-val-c))))

(defmethod make-filter-options :supervised-nominal-to-binary
  ([kind map]
     (let [cols-val (check-options {:also-binary "-N"
                                    :for-each-nominal "-A"}
                                   map
                                   [""])]
    (into-array cols-val))))

(defmethod make-filter-options :unsupervised-nominal-to-binary
  ([kind map]
     (let [cols (get map :attributes)
           pre-cols (reduce #(str %1 "," (+ %2 1)) "" cols)
           cols-val-a ["-R" (.substring pre-cols 1 (.length pre-cols))]
           cols-val-b (check-options {:invert "-V"
                                      :also-binary "-N"
                                      :for-each-nominal "-A"}
                                     map
                                     cols-val-a)]
       (into-array cols-val-b))))


;; Creation of filters

(defmacro make-filter-m [kind options filter-class]
  `(let [filter# (new ~filter-class)
         dataset# (get ~options :dataset)
         opts# (make-filter-options ~kind ~options)]
     (.setOptions filter# opts#)
     (.setInputFormat filter# dataset#)
     filter#))

(defmulti make-filter
  "Creates a filter for datasets"
  (fn [kind options] kind))

(defmethod make-filter :supervised-discretize
  ([kind options]
     (make-filter-m kind options weka.filters.supervised.attribute.Discretize)))


(defmethod make-filter :unsupervised-discretize
  ([kind options]
     (make-filter-m kind options weka.filters.unsupervised.attribute.Discretize)))

(defmethod make-filter :supervised-nominal-to-binary
  ([kind options]
     (make-filter-m kind options weka.filters.supervised.attribute.NominalToBinary)))

(defmethod make-filter :unsupervised-nominal-to-binary
    ([kind options]
     (make-filter-m kind options weka.filters.unsupervised.attribute.NominalToBinary)))

;; Processing the filtering of data

(defn filter-process
  "Filters an input dataset using the provided filter and generates an output dataset"
  [filter dataset]
  (Filter/useFilter dataset filter))
