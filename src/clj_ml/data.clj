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

(declare dataset-seq)

;; Common functions

(defn is-instance?
  "Checks if the provided object is an instance"
  [instance]
  (instance? weka.core.Instance instance))

(defn is-dataset?
  "Checks if the provided object is a dataset"
  [dataset]
  (instance? weka.core.Instances dataset))


(defn index-attr
  "Returns the index of an attribute in the attributes definition of an
   instance or dataset"
  [dataset attr-name]
  (if (number? attr-name)
    attr-name
    (let [attr-name (name attr-name)]
      (find-first #(= attr-name (.name (.attribute dataset %))) (range (.numAttributes dataset))))))

(defn attribute-at
  "Returns attribute situated at the provided position or the provided name."
  [dataset-or-instance index-or-name]
  (.attribute dataset-or-instance (index-attr dataset-or-instance index-or-name)))

(defn attribute-name-at
  "Returns the name of an attribute situated at the provided position in
   the attributes definition of an instance or class"
  [dataset-or-instance index-or-name]
  (let [^Attribute class-attr (attribute-at dataset-or-instance index-or-name)]
    (.name class-attr)))

(defn attributes
  "Returns the attributes (weka.core.Attribute) of the dataset or instance"
  [dataset-or-instance]
  (map #(.attribute dataset-or-instance %) (range (.numAttributes dataset-or-instance))))

(defn keyword-name [^Attribute attr]
  (keyword (.name attr)))

(defn attribute-names
  "Returns the attribute names, as keywords, of the dataset or instance"
  [dataset-or-instance]
  (map keyword-name (attributes dataset-or-instance)))

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
  "Creates a nominal weka.core.Attribute with the given name and labels"
  [attr-name labels]
  (Attribute. (name attr-name) (into-fast-vector (map name labels))))

(defn dataset-index-attr [dataset attr]
  (index-attr dataset attr))

(defn instance-index-attr [instance attr]
  (index-attr instance attr))

;; Construction of individual data and datasets

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
                   (.setValue inst c (name (first vs)))
                   (if (sequential? (first vs))
                     ;; this is a map of labels
                     (let [k (name (nth (first vs) 0))
                           val (nth (first vs) 1)
                           ik  (index-attr inst k)]
                       (if (or (keyword? val) (string? val))
                         ;; this is a nominal entry in keyword or string form
                         (.setValue inst ik (name val))
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
                                    vfa (reduce (fn [a i] (.addElement a (name i)) a)
                                                (new FastVector) v)]
                                (new Attribute (name (first (keys att))) vfa))
                              (new Attribute (name (first (keys att))) (first (vals att))))
                            (new Attribute (name att)))))
           (recur (rest atts)
                  fv))))))

(defn make-dataset
  "Creates a new dataset, empty or with the provided instances and options"
  ([ds-name attributes capacity-or-labels & opts]
     (let [options (first-or-default opts {})
           weight (get options :weight 1)
           class-attribute (get options :class)
           ds (if (sequential? capacity-or-labels)
                ;; we have received a sequence instead of a number, so we initialize data
                ;; instances in the dataset
                (let [dataset (new ClojureInstances (name ds-name) (parse-attributes attributes) (count capacity-or-labels))]
                  (loop [vs capacity-or-labels]
                    (if (empty? vs)
                      dataset
                      (do
                        (let [inst (make-instance dataset weight (first vs))]
                          (.add dataset inst))
                        (recur (rest vs))))))
                ;; we haven't received a vector so we create an empty dataset
                (new Instances (name ds-name) (parse-attributes attributes) capacity-or-labels))]
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


(defn dataset-labels-at [dataset-or-instance index-or-name]
  "Returns the lables (possible values) for a nominal attribute at the provided position"
  (let [^Attribute attr (attribute-at dataset-or-instance index-or-name)
        values (enumeration-seq (.enumerateValues attr))]
    (if (empty? values)
      :not-nominal
      (reduce (fn [m ^String val]
                (assoc m (keyword val) (.indexOfValue attr val)))
              {}
              values))))

(defn dataset-class-labels
  "Returns the possible labels for the class attribute"
  [^Instances dataset]
  (dataset-labels-at dataset (.classIndex dataset)))

(defn dataset-format
  "Returns the definition of the attributes of this dataset"
  [dataset]
   (reduce
    (fn [so-far ^Attribute attr]
      (conj so-far
            (if (.isNominal attr)
              {(keyword-name attr) (map keyword (enumeration-seq (.enumerateValues attr)))}
              (keyword-name attr))))
    []
    (attributes dataset)))

(defn dataset-get-class
  "Returns the index of the class attribute for this dataset"
  [^Instances dataset]
  (.classIndex dataset))

(defn dataset-nominal?
  "Returns boolean indicating if the class attribute is nominal"
  [^Instances dataset]
  (.. dataset classAttribute isNominal))

(defn dataset-class-values
  "Returns a lazy-seq of the values for the dataset's class attribute.
If the class is nominal then the string value (not keyword) is returned."
  [^Instances dataset]
  (let [class-attr (.classAttribute dataset)
        class-value (if (.isNominal class-attr)
                      (fn [^Instance i] (.stringValue i class-attr))
                      (fn [^Instance i] (.classValue i)))] ;classValue returns the double
    (map class-value (dataset-seq dataset))))

;; manipulation of instances

(defn instance-set-class
  "Sets the index of the class attribute for this instance"
  [^Instance instance pos]
  (doto instance (.setClassValue (int pos))))

(defn instance-get-class
  "Get the index of the class attribute for this instance"
   [^Instance instance]
  (.classValue instance))

(defn instance-value-at
  "Returns the value of an instance attribute. A string, not a keyword is returned."
  [^Instance instance pos]
  (let [pos (int pos)
        attr (.attribute instance pos)]
    (if (.isNominal attr) ; This ignores the fact that weka can have date and other attribute types...
      (.stringValue instance pos)
      (.value instance pos))))

(defn instance-to-list
  "Builds a list with the values of the instance"
  [^Instance instance]
  (map (partial instance-value-at instance) (range (.numValues instance))))

(defn instance-to-vector
  "Builds a vector with the values of the instance"
  [instance]
  (vec (instance-to-list instance)))

(defn instance-to-map
  "Builds a vector with the values of the instance"
  [^Instance instance]
  (reduce (fn [m i]
            (assoc m (keyword (attribute-name-at instance i)) (instance-value-at instance i)))
          {}
          (range (.numValues instance))))


;; manipulation of datasets

(defn dataset-seq
  "Builds a new clojure sequence from this dataset"
  [dataset]
  (if (= (class dataset)
         ClojureInstances)
    (seq dataset)
    (seq (new ClojureInstances ^Instances dataset))))

(defn dataset-as-maps
  "Returns a lazy sequence of the dataset represetned as maps.
This fn is preferale to mapping over a seq yourself with instance-to-map
becuase it avoids redundant string interning of the attribute names."
  [dataset]
  (let [attrs (attribute-names dataset)] ; we only want to intern the attribute names once!
    (for [instance (map instance-to-list (dataset-seq dataset))]
      (zipmap attrs instance))))

(defn dataset-set-class
  "Sets the index of the attribute of the dataset that is the class of the dataset"
  [^Instances dataset index-or-name]
  (doto dataset (.setClassIndex ^int (index-attr dataset index-or-name))))

(defn dataset-remove-class
  "Removes the class attribute from the dataset"
  [^Instances dataset]
  (doto dataset (.setClassIndex -1)))

(defn dataset-count
  "Returns the number of elements in a dataset"
  [^Instances dataset]
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

(defn dataset-replace-attribute!
  "Replaces the specified attribute with the given one. (The attribute should be a weka.core.Attribute)
This function only modifies the format of the dataset and does not deal with any instances.
The intention is for this to be used on data-formats and not on datasets with data."
  [dataset attr-name new-attr]
  (let [attr-pos (index-attr dataset attr-name)]
    (doto dataset
      (.deleteAttributeAt attr-pos)
      (.insertAttributeAt new-attr attr-pos))))
