;;
;; Clusterers
;; @author Antonio Garrote
;;

(ns #^{:author "Antonio Garrote <antoniogarrote@gmail.com>"}
  clj-ml.clusterers
  "This namespace contains several functions for
   building clusterers using different clustering algorithms. K-means, Cobweb and

   Expectation maximization algorithms are currently supported. Some of these
   algorithms support incremental building of the clustering without having the
   full data set in main memory. Functions for evaluating the clusterer as well
   as for clustering new instances are also supported"
  (:use [clj-ml utils data distance-functions]
        [incanter charts])
  (:import (java.util Date Random)
           (weka.clusterers ClusterEvaluation SimpleKMeans Cobweb EM)))


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


(defmethod make-clusterer-options :cobweb
  ([kind map]
     (let [cols-val-a (check-option-values {:acuity "-A"
                                            :cutoff "-C"
                                            :random-seed "-S"}
                                           map
                                           [""])]
    (into-array cols-val-a))))


(defmethod make-clusterer-options :expectation-maximization
  ([kind map]
     (let [cols-val-a (check-option-values {:acuity "-A"
                                            :cutoff "-C"
                                            :random-seed "-S"}
                                           map
                                           [""])]
    (into-array cols-val-a))))


;; Building clusterers

(defmacro make-clusterer-m
  ([kind clusterer-class options]
     `(let [options-read# (if (empty? ~options)  {} (first ~options))
            clusterer# (new ~clusterer-class)
            opts# (make-clusterer-options ~kind options-read#)]
        (.setOptions clusterer# opts#)
        (when (not (empty? (get options-read# :distance-function)))
          ;; We have to setup a different distance function
          (let [dist# (get options-read# :distance-function)
                real-dist# (if (map? dist#)
                             (make-distance-function (first (keys dist#))
                                                     (first (vals dist#)))
                             dist#)]
            (.setDistanceFunction clusterer# real-dist#)))
        clusterer#)))

(defmulti make-clusterer
  "Creates a new clusterer for the given kind algorithm and options"
  (fn [kind & options] kind))


(defmethod make-clusterer :k-means
  ([kind & options]
     (make-clusterer-m kind SimpleKMeans options)))

(defmethod make-clusterer :cobweb
  ([kind & options]
     (make-clusterer-m kind Cobweb options)))

(defmethod make-clusterer :expectation-maximization
  ([kind & options]
     (make-clusterer-m kind EM options)))


;; Clustering data

(defn clusterer-build
  "Applies a clustering algorithm to a set of data"
  ([clusterer dataset]
     (.buildClusterer clusterer dataset)))

(defn clusterer-update
  "If the clusterer is updateable it updates the cluster with the given instance or set of instances"
  ([clusterer instance-s]
     (if (is-dataset? instance-s)
       (do (for [i (dataset-seq instance-s)]
             (.updateClusterer clusterer i))
           (.updateFinished clusterer)
           clusterer)
       (do (.updateClusterer clusterer instance-s)
           (.updateFinished clusterer)
           clusterer))))

;; Retrieving information from a clusterer

(defmulti clusterer-info
  "Retrieves the data from a cluster, these data are clustering-algorithm dependent"
  (fn [clusterer] (class clusterer)))

(defmethod clusterer-info SimpleKMeans
  ([clusterer]
     "Accepts a k-means clusterer
      Returns a map with:
       :number-clusters The number of clusters in the clusterer
       :centroids       Map with the identifier and the centroid values for each cluster
       :cluster-sizes   Number of data points classified in each cluster
       :squared-error   Minimized squared error"
     {:number-clusters (.numberOfClusters clusterer)
      :centroids (second
                  (reduce (fn [acum item]
                            (let [counter (first acum)
                                  map     (second acum)]
                              (list (+ counter 1)
                                    (conj map {counter item}))))
                          (list 0 {})
                          (dataset-seq (.getClusterCentroids clusterer))))
      :cluster-sizes (let [sizes (.getClusterSizes clusterer)]
                       (reduce (fn [acum item]
                                 (conj acum {item (aget sizes item)}))
                               {}
                               (range 0 (.numberOfClusters clusterer))))
      :squared-error (.getSquaredError clusterer)}))



;; Evaluating clusterers

(defn- collect-evaluation-results
  "Collects all the statistics from the evaluation of a clusterer"
  ([evaluation]
     (do
       (println "hola?")
       (println (.clusterResultsToString evaluation))
       {:classes-to-clusters (try-metric
                              #(reduce (fn [acum i] (conj acum {i (aget (.getClassesToClusters evaluation) i)}))
                                       {}
                                       (range 0 (.getNumClusters evaluation))))
        :log-likelihood (try-metric #(.getLogLikelihood evaluation))
        :evaluation-object evaluation})))


(defmulti clusterer-evaluate
  "Evaluates a trained clusterer using the provided dataset or cross-validation"
  (fn [clusterer mode & evaluation-data] mode))

(defmethod clusterer-evaluate :dataset
  ([clusterer mode & evaluation-data]
     (let [test-data (nth evaluation-data 0)
           evaluation (do (let [evl (new ClusterEvaluation)]
                            (.setClusterer evl clusterer)
                            evl))]
       (.evaluateClusterer evaluation test-data)
       (println (.clusterResultsToString evaluation))
       (collect-evaluation-results evaluation))))

(defmethod clusterer-evaluate :cross-validation
  ([clusterer mode & evaluation-data]
     (let [training-data (nth evaluation-data 0)
           folds (nth evaluation-data 1)
           evaluation (let [evl (new ClusterEvaluation)]
                        (.setClusterer evl clusterer)
                        evl)
           log-likelihood (ClusterEvaluation/crossValidateModel clusterer
                                                                training-data
                                                                folds
                                                                (new Random (.getTime (new Date))))]
     {:log-likelihood log-likelihood})))


;; Clustering collections

(defn clusterer-cluster
  "Add a class to each instance according to the provided clusterer"
  ([clusterer dataset]
     (let [attributes (conj (clj-ml.data/dataset-format dataset)
                            {:class (map #(keyword (str %1)) (range 0 (.numberOfClusters clusterer)))})
           clustered (map (fn [i] (conj (instance-to-vector i)
                                        (keyword (str (.clusterInstance clusterer i)))))
                          (dataset-seq dataset))
           nds (make-dataset (keyword (str "clustered " (dataset-name dataset)))
                             attributes
                             clustered)]
       (dataset-set-class nds (- (count attributes) 1))
       nds)))
