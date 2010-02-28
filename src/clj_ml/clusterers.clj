;;
;; Clusterers
;; @author Antonio Garrote
;;

(ns clj-ml.clusterers
  (:use [clj-ml utils data])
  (:import (java.util Date Random)
           (weka.clusterers SimpleKMeans)))

;; Setting up clusterer options

(defmulti make-clusterer-options
  "Creates ther right parameters for a clusterer"
  (fn [kind map] kind))

(defmethod make-clusterer-options :k-means
  ([kind map]
     (let [cols-val (check-options {:display-standard-deviation "-V"
                                    :replace-missing-values "-M"
                                    :preserve-instances-order "-O"}
                                     map
                                     [""])
           cols-val-a (check-option-values {:number-clusters "-N"
                                            :random-seed "-S"
                                            :number-iterations "-I"}
                                           map
                                           cols-val)]
    (into-array cols-val-a))))

;; Building clusterers

(defmacro make-clusterer-m
  ([kind clusterer-class options]
     `(let [options-read# (if (empty? ~options)  {} (first ~options))
            clusterer# (new ~clusterer-class)
            opts# (make-clusterer-options ~kind options-read#)]
        (.setOptions clusterer# opts#)
        clusterer#)))

(defmulti make-clusterer
  "Creates a new clusterer for the given kind algorithm and options"
  (fn [kind & options] kind))


(defmethod make-clusterer :k-means
  ([kind & options]
     (make-clusterer-m kind SimpleKMeans options)))


;; Clustering data

(defn clusterer-build
  "Applies a clustering algorithm to a set of data"
  ([clusterer dataset]
     (.buildClusterer clusterer dataset)))
