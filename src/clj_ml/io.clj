;;
;; Storing and reading data from different formats
;; @author Antonio Garrote
;;

(ns #^{:author "Antonio Garrote <antoniogarrote@gmail.com>"}
  clj-ml.io
  "Functions for reading and saving datasets, classifiers and clusterers to files and other
   persistence mechanisms."
  (:import (weka.core.converters CSVLoader ArffLoader XRFFLoader)
           (weka.core.converters CSVSaver ArffSaver XRFFSaver)
           (java.io File)
           (java.net URL URI)))


;; Loading of instances

(defmulti load-instances
  "Load instances from different data sources"
  (fn [kind source & options] kind))

(defmacro m-load-instances [loader source]
  `(do
     (if (= (class ~source) java.lang.String)
       (.setSource ~loader (new URL ~source))
       (if (= (class ~source) java.io.File)
         (.setFile ~loader ~source)))
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
     (if (= (class ~destiny) java.lang.String)
       (.setFile ~saver (new File (new URI ~destiny)))
       (if (= (class ~destiny) java.io.File)
         (.setFile ~saver ~destiny)))
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
