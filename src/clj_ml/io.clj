;;
;; Storing and reading data from different formats
;; @author Antonio Garrote
;;

(ns clj-ml.io
  (:import (weka.core.converters CSVLoader ArffLoader XRFFLoader)
           (weka.core.converters CSVSaver ArffSaver XRFFSaver)
           (java.io File)
           (java.net URL URI)))


;; Loading of instances

(defmulti load-instances
  "Load instances from different data sources"
  (fn [kind source] kind))

(defmacro m-load-instances [loader source]
  `(do
     (if (= (class ~source) java.lang.String)
       (.setSource ~loader (new URL ~source))
       (if (= (class ~source) java.io.File)
         (.setFile ~loader ~source)))
     (.getDataSet ~loader)))

(defmethod load-instances :arff
  ([kind source]
     (let [loader (new ArffLoader)]
       (m-load-instances loader source))))


(defmethod load-instances :xrff
  ([kind source]
     (let [loader (new XRFFLoader)]
       (m-load-instances loader source))))

(defmethod load-instances :csv
  ([kind source]
     (let [loader (new CSVLoader)]
       (m-load-instances loader source))))


;; Saving of instances

(defmulti save-instances
  "Save instances into data destinies"
  (fn [kind destiny instances] kind))

(defmacro m-save-instances [saver destiny instances]
  `(do
     (if (= (class ~destiny) java.lang.String)
       (.setFile ~saver (new File (new URI ~destiny)))
       (if (= (class ~destiny) java.io.File)
         (.setFile ~saver ~destiny)))
     (.setInstances ~saver ~instances)
     (.writeBatch ~saver)))

(defmethod save-instances :arff
  ([kind destiny instances]
     (let [saver (new ArffSaver)]
       (m-save-instances saver destiny instances))))

(defmethod save-instances :xrff
  ([kind destiny instances]
     (let [saver (new XRFFSaver)]
       (m-save-instances saver destiny instances))))

(defmethod save-instances :csv
  ([kind destiny instances]
     (let [saver (new CSVSaver)]
       (m-save-instances saver destiny instances))))

