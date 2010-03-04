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

(defmethod make-filter-options :remove-attributes
  ([kind map]
     (let [cols (get map :attributes)
           pre-cols (reduce #(str %1 "," (+ %2 1)) "" cols)
           cols-val-a ["-R" (.substring pre-cols 1 (.length pre-cols))]
           cols-val-b (check-options {:invert "-V"}
                                     map
                                     cols-val-a)]
       (into-array cols-val-b))))

(defmethod make-filter-options :select-append-attributes
  ([kind map]
     (let [cols (get map :attributes)
           pre-cols (reduce #(str %1 "," (+ %2 1)) "" cols)
           cols-val-a ["-R" (.substring pre-cols 1 (.length pre-cols))]
           cols-val-b (check-options {:invert "-V"}
                                     map
                                     cols-val-a)]
       (into-array cols-val-b))))

(defmethod make-filter-options :project-attributes
  ([kind options]
     (let [opts (if (nil? (:invert options))
                  (conj options {:invert true})
                  (dissoc options :invert))]
       (make-filter-options :remove-attributes opts))))


;; Creation of filters

(defmacro make-filter-m [kind options filter-class]
  `(let [filter# (new ~filter-class)
         dataset-format# (get ~options :dataset-format)
         opts# (make-filter-options ~kind ~options)]
     (.setOptions filter# opts#)
     (.setInputFormat filter# dataset-format#)
     filter#))

(defmulti make-filter
  "Creates a filter for the provided attributes format"
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

(defmethod make-filter :remove-attributes
  ([kind options]
     (make-filter-m kind options weka.filters.unsupervised.attribute.Remove)))

(defmethod make-filter :select-append-attributes
  ([kind options]
     (make-filter-m kind options weka.filters.unsupervised.attribute.Copy)))

(defmethod make-filter :project-attributes
  ([kind options]
     (make-filter-m kind options weka.filters.unsupervised.attribute.Remove)))

;; Processing the filtering of data

(defn filter-apply
  "Filters an input dataset using the provided filter and generates an output dataset"
  [filter dataset]
  (Filter/useFilter dataset filter))

(defn make-apply-filter
  "Creates a new filter with the provided options and apply it to the provided dataset.
   The dataset-format attribute for the making of the filter will be setup to the
   dataset passed as an argument if no other valu is provided"
  [kind options dataset]
  (let [opts (if (nil? (:dataset-format options)) (conj options {:dataset-format dataset}))
        filter (make-filter kind opts)
        _foo (println (str "Options are " opts))]
    (filter-apply filter dataset)))
