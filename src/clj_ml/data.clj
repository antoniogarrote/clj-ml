;;
;; Manipulation of datasets and instances
;; @author Antonio Garrote
;;

(ns #^{:author "Antonio Garrote <antoniogarrote@gmail.com>"}
  clj-ml.data
  "This namespace contains several functions for
   building creating and manipulating data sets and instances. The formats of
   these data sets as well as their classes can be modified and assigned to
   the instances. Finally data sets can be transformed into Clojure sequences
   that can be transformed using usual Clojure functions like map, reduce, etc."
  (:use [clj-ml utils]
        [clojure.contrib.seq :only [find-first]])
  (:import (weka.core Instance Instances FastVector Attribute)
           (cljml ClojureInstances)))

;; Common functions

(defn is-instance?
  "Checks if the provided object is an instance"
  [instance]
  (instance? weka.core.Instance instance))

(defn is-dataset?
  "Checks if the provided object is a dataset"
  [dataset]
  (instance? weka.core.Instances dataset))

;; Construction of individual data and datasets

(defn attribute-name-at
  "Returns the name of an attribute situated at the provided position in
   the attributes definition of an instance or class"
  [dataset-or-instance pos]
  (let [class-attr (.attribute dataset-or-instance pos)]
    (.name class-attr)))

(defn index-attr
  "Returns the index of an attribute in the attributes definition of an
   instance or dataset"
  [dataset attr-name]
  (let [attr-name (name attr-name)]
    (find-first #(= attr-name (.name (.attribute dataset %))) (range (.numAttributes dataset)))))

(defn attributes
  "Returns the attributes (weka.core.Attribute) of the dataset or instance"
  [dataset-or-instance]
  (map #(.attribute dataset-or-instance %) (range (.numAttributes dataset-or-instance))))

(defn attribute-names
  "Returns the attribute names, as keywords, of the dataset or instance"
  [dataset-or-instance]
  (map (comp keyword #(.name %)) (attributes dataset-or-instance)))

(defn numeric-attributes
  "Returns the numeric attributes (weka.core.Attribute) of the dataset or instance"
  [dataset-or-instance]
  (filter #(.isNumeric %) (attributes dataset-or-instance)))

(defn nominal-attributes
  "Returns the string attributes (weka.core.Attribute) of the dataset or instance"
  [dataset-or-instance]
  (filter #(.isNominal %) (attributes dataset-or-instance)))

(defn string-attributes
  "Returns the string attributes (weka.core.Attribute) of the dataset or instance"
  [dataset-or-instance]
  (filter #(.isString %) (attributes dataset-or-instance)))

(defn nominal-attribute
  "Creates a nominal weka.core.Attribute with the given name and values"
  [attr-name values]
  (Attribute. (name attr-name) (into-fast-vector (map name values))))

(defn dataset-index-attr [dataset attr]
  (index-attr dataset attr))

(defn instance-index-attr [instance attr]
  (index-attr instance attr))


(defn make-instance
  "Creates a new dataset instance from a vector"
  ([dataset vector]
     (make-instance dataset 1 vector))
  ([dataset weight vector]
     (let [inst (new Instance
                     (count vector))]
       (do (.setDataset inst dataset)
           (loop [vs vector
                  c 0]
             (if (empty? vs)
               (doto inst (.setWeight (double weight)))
               (do
                 (if (or (keyword? (first vs)) (string? (first vs)))
                   ;; this is a nominal entry in keyword or string form
                   (.setValue inst c (key-to-str (first vs)))
                   (if (sequential? (first vs))
                     ;; this is a map of values
                     (let [k (key-to-str (nth (first vs) 0))
                           val (nth (first vs) 1)
                           ik  (index-attr inst k)]
                       (if (or (keyword? val) (string? val))
                         ;; this is a nominal entry in keyword or string form
                         (.setValue inst ik (key-to-str val))
                         (.setValue inst ik (double val))))
                     ;; A double value for the entry
                     (.setValue inst c (double (first vs)))))
                 (recur (rest vs)
                        (+ c 1)))))))))


(defn- parse-attributes
  "Builds a set of attributes for a dataset parsed from the given array"
  ([attributes]
     (loop [atts attributes
            fv (new FastVector (count attributes))]
       (if (empty? atts)
         fv
         (do
           (let [att (first atts)]
             (.addElement fv
                          (if (map? att)
                            (if (sequential? (first (vals att)))
                              (let [v (first (vals att))
                                    vfa (reduce (fn [a i] (.addElement a (key-to-str i)) a)
                                                (new FastVector) v)]
                                (new Attribute (key-to-str (first (keys att))) vfa))
                              (new Attribute (key-to-str (first (keys att))) (first (vals att))))
                            (new Attribute (key-to-str att)))))
           (recur (rest atts)
                  fv))))))

(defn make-dataset
  "Creates a new dataset, empty or with the provided instances and options"
  ([name attributes capacity-or-values & opts]
     (let [options (first-or-default opts {})
           weight (get options :weight 1)
           class-attribute (get options :class)
           ds (if (sequential? capacity-or-values)
                ;; we have received a sequence instead of a number, so we initialize data
                ;; instances in the dataset
                (let [dataset (new ClojureInstances (key-to-str name) (parse-attributes attributes) (count capacity-or-values))]
                  (loop [vs capacity-or-values]
                    (if (empty? vs)
                      dataset
                      (do
                        (let [inst (make-instance dataset weight (first vs))]
                          (.add dataset inst))
                        (recur (rest vs))))))
                ;; we haven't received a vector so we create an empty dataset
                (new Instances (key-to-str name) (parse-attributes attributes) capacity-or-values))]
       ;; we try to setup the class attribute if :class with a attribute name or
       ;; integer value is provided
       (when (not (nil? class-attribute))
         (let [index-class-attribute (if (keyword? class-attribute)
                                       (loop [c 0
                                              acum attributes]
                                           (if (= (let [at (first acum)]
                                                        (if (map? at)
                                                          (first (keys at))
                                                          at))
                                                  class-attribute)
                                             c
                                             (if (= c (count attributes))
                                               (throw (new Exception "provided class attribute not found"))
                                               (recur (+ c 1)
                                                      (rest acum)))))
                                           class-attribute)]
           (.setClassIndex ds index-class-attribute)))
       ds)))

;; dataset information

(defn dataset-name
  "Returns the name of this dataset"
  [dataset]
  (.relationName dataset))

(defn dataset-class-values
  "Returns the possible values for the class attribute"
  [dataset]
  (let [class-attr (.classAttribute dataset)
        values (.enumerateValues class-attr)]
    (loop [continue (.hasMoreElements values)
           acum {}]
      (if continue
        (let [val (.nextElement values)
              index (.indexOfValue class-attr val)]
          (recur (.hasMoreElements values)
                 (conj acum {(keyword val) index})))
        acum))))

(defn dataset-values-at [dataset-or-instance pos]
  "Returns the possible values for a nominal attribute at the provided position"
  (let [class-attr (.attribute dataset-or-instance pos)
        values (.enumerateValues class-attr)]
    (if (nil? values)
      :not-nominal
      (loop [continue (.hasMoreElements values)
             acum {}]
        (if continue
          (let [val (.nextElement values)
                index (.indexOfValue class-attr val)]
            (recur (.hasMoreElements values)
                   (conj acum {(keyword val) index})))
          acum)))))

(defn dataset-format
  "Returns the definition of the attributes of this dataset"
  [dataset]
  (let [max (.numAttributes dataset)]
    (loop [acum []
           c 0]
      (if (< c max)
        (let [attr (.attribute dataset c)
              index c
              name (keyword (.name attr))
              nominal? (.isNominal attr)
              to-add (if nominal?
                       (let [vals (dataset-values-at dataset index)]
                         {name (keys vals)})
                       name)]
          (recur (conj acum to-add)
                 (+ c 1)))
        acum))))

(defn dataset-get-class
  "Returns the index of the class attribute for this dataset"
  [dataset]
  (.classIndex dataset))

;; manipulation of instances

(defn instance-set-class
  "Sets the index of the class attribute for this instance"
  [instance pos]
  (doto instance (.setClassValue pos)))

(defn instance-get-class
  "Get the index of the class attribute for this instance"
   [instance]
  (.classValue instance))

(defn instance-value-at
  "Returns the value of an instance attribute"
  [instance pos]
  (let [attr (.attribute instance pos)]
    (if (.isNominal attr)
      (let [val (.value instance pos)
            key-vals (dataset-values-at instance pos)
            key-val (loop [ks (keys key-vals)]
                      (if (= (get key-vals (first ks))
                             val)
                        (first ks)
                        (recur (rest ks))))]
        key-val)
      (.value instance pos))))

(defn instance-to-vector
  "Builds a vector with the values of the instance"
  [instance]
  (let [max (.numValues instance)]
    (loop [c 0
           acum []]
      (if (= c max)
        acum
        (recur (+ c 1)
               (conj acum (instance-value-at instance c)))))))

(defn instance-to-map
  "Builds a vector with the values of the instance"
  [instance]
  (let [max (.numValues instance)]
    (loop [c 0
           acum {}]
      (if (= c max)
        acum
        (recur (+ c 1)
               (conj acum {(keyword (. (.attribute instance c) name))
                           (instance-value-at instance c)} ))))))


;; manipulation of datasets

(defn dataset-seq
  "Builds a new clojure sequence from this dataset"
  [dataset]
  (if (= (class dataset)
         ClojureInstances)
    (seq dataset)
    (seq (new ClojureInstances dataset))))

(defn dataset-as-maps
  "Returns a lazy sequence of the dataset represetned as maps."
  [dataset]
  (map instance-to-map (dataset-seq dataset)))

(defn dataset-set-class
  "Sets the index of the attribute of the dataset that is the class of the dataset"
  [dataset pos]
  (doto dataset (.setClassIndex pos)))

(defn dataset-remove-class
  "Removes the class attribute from the dataset"
  [dataset]
  (doto dataset (.setClassIndex -1)))

(defn dataset-count
  "Returns the number of elements in a dataset"
  [dataset]
  (.numInstances dataset))

(defn dataset-add
  "Adds a new instance to a dataset. A clojure vector or an Instance
   can be passed as arguments"
  ([dataset vector]
     (dataset-add dataset 1 vector))
  ([dataset weight vector]
     (do
       (if (= (class vector) weka.core.Instance)
         (.add dataset vector)
         (let [instance (make-instance dataset weight vector)]
           (.add dataset instance)))
       dataset)))

(defn dataset-extract-at
  "Removes and returns the instance at a certain position from the dataset"
  [dataset pos]
  (let [inst (.instance dataset pos)]
    (do
      (.delete dataset pos)
      inst)))

(defn dataset-at
  "Returns the instance at a certain position from the dataset"
  [dataset pos]
  (.instance dataset pos))

(defn dataset-pop
  "Removes and returns the first instance in the dataset"
  [dataset]
  (dataset-extract-at dataset 0))
