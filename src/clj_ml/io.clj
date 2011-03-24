;;
;; Storing and reading data from different formats
;; @author Antonio Garrote
;;

(ns #^{:author "Antonio Garrote <antoniogarrote@gmail.com>"}
  clj-ml.io
  "Functions for reading and saving datasets, classifiers and clusterers to files and other
   persistence mechanisms."
  (:require clj-ml.data-store)
  (:import (weka.core.converters CSVLoader ArffLoader XRFFLoader)
           (weka.core.converters CSVSaver ArffSaver XRFFSaver)
           (java.io File InputStream OutputStream)
           (java.net URL URI)))


;; Loading of instances

(defmulti load-instances
  "Load instances from different data sources"
  (fn [kind source & options] kind))

(defmacro m-load-instances [loader source]
  `(do
     (if (= (class ~source)  java.io.File)
       (.setFile ~loader ~source)
       (.setSource ~loader (if (= (class ~source) java.lang.String)
                             (new URL ~source)
                             ^InputStream ~source)))
     (.getDataSet ~loader)))

(defmethod load-instances :arff
  ([kind source & options]
     (let [loader (new ArffLoader)]
       (m-load-instances loader source))))


(defmethod load-instances :xrff
  ([kind source & options]
     (let [loader (new XRFFLoader)]
       (m-load-instances loader source))))

(defmethod load-instances :csv
  ([kind source & options]
     (let [loader (new CSVLoader)]
       (m-load-instances loader source))))

(defmethod load-instances :mongodb
  ([kind source & options]
     (let [database {:database source}
           name {:dataset-name source}]
       (clj-ml.data-store/data-store-load-dataset :mongodb database name options))))

;; Saving of instances

(defmulti save-instances
  "Save instances into data destinies"
  (fn [kind destiny instances & options] kind))

(defmacro m-save-instances [saver destiny instances]
  `(do
     (prn ~destiny)
     (condp #(isa? %2 %1) (class ~destiny)
         String (do (println "I'm a string!") (.setFile ~saver (File. (URI. ~destiny))))
         File (do (println "I'm a file!") (.setFile ~saver ~destiny))
         OutputStream (do (println "I'm a stream!") (.setDestination ~saver ~destiny)))
     (.setInstances ~saver ~instances)
     (.writeBatch ~saver)))

(defmethod save-instances :arff
  ([kind destiny instances & options]
     (let [saver (new ArffSaver)]
       (m-save-instances saver destiny instances))))

(defmethod save-instances :xrff
  ([kind destiny instances & options]
     (let [saver (new XRFFSaver)]
       (m-save-instances saver destiny instances))))

(defmethod save-instances :csv
  ([kind destiny instances & options]
     (let [saver (new CSVSaver)]
       (m-save-instances saver destiny instances))))

(defmethod save-instances :mongodb
  ([kind destiny instances & options]
     (clj-ml.data-store/data-store-save-dataset :mongodb destiny instances options)))
