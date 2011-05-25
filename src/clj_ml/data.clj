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
(declare instance-index-attr dataset-index-attr)

;; Common functions

(defn is-instance?
  "Checks if the provided object is an instance"
  [instance]
  (instance? weka.core.Instance instance))

(defn is-dataset?
  "Checks if the provided object is a dataset"
  [dataset]
  (instance? weka.core.Instances dataset))


(defn instance-attribute-at [^Instance instance index-or-name]
  (.attribute instance (int (instance-index-attr instance index-or-name))))

(defn dataset-attribute-at [^Instances dataset index-or-name]
  (.attribute dataset (int (dataset-index-attr dataset index-or-name))))

(defn attribute-at
  "Returns attribute situated at the provided position or the provided name."
  [dataset-or-instance index-or-name]
  (if (is-instance? dataset-or-instance)
    (instance-attribute-at dataset-or-instance index-or-name)
    (dataset-attribute-at dataset-or-instance index-or-name)))

(defn attribute-name-at
  "Returns the name of an attribute situated at the provided position in
   the attributes definition of an instance or class"
  [dataset-or-instance index-or-name]
  (let [^Attribute class-attr (attribute-at dataset-or-instance index-or-name)]
    (.name class-attr)))

(defn dataset-attributes
  "Returns the attributes (weka.core.Attribute) of the dataset or instance"
  [^Instances dataset]
  (map #(.attribute dataset (int %)) (range (.numAttributes dataset))))

(defn instance-attributes
  "Returns the attributes (weka.core.Attribute) of the dataset or instance"
  [^Instance instance]
  (map #(.attribute instance (int %)) (range (.numAttributes instance))))

(defn attributes
  "Returns the attributes (weka.core.Attribute) of the dataset or instance"
  [dataset-or-instance]
  (if (is-instance? dataset-or-instance)
    (instance-attributes dataset-or-instance)
    (dataset-attributes dataset-or-instance)))

(defn attr-name [^Attribute attr]
  (.name attr))

(defn keyword-name [attr]
  (keyword (attr-name attr)))

(defn attribute-names
  "Returns the attribute names, as keywords, of the dataset or instance"
  [dataset-or-instance]
  (map keyword-name (attributes dataset-or-instance)))

(defn numeric-attributes
  "Returns the numeric attributes (weka.core.Attribute) of the dataset or instance"
  [dataset-or-instance]
  (filter #(.isNumeric ^Attribute %) (attributes dataset-or-instance)))

(defn nominal-attributes
  "Returns the string attributes (weka.core.Attribute) of the dataset or instance"
  [dataset-or-instance]
  (filter #(.isNominal ^Attribute %) (attributes dataset-or-instance)))

(defn string-attributes
  "Returns the string attributes (weka.core.Attribute) of the dataset or instance"
  [dataset-or-instance]
  (filter #(.isString ^Attribute %) (attributes dataset-or-instance)))

(defn nominal-attribute
  "Creates a nominal weka.core.Attribute with the given name and labels"
  [attr-name labels]
  (Attribute. ^String (name attr-name) ^FastVector (into-fast-vector (map name labels))))

(defn dataset-index-attr
  "Returns the index of an attribute in the attributes definition of a dataset."
  [^Instances dataset attr]
  (if (number? attr)
    attr
    (find-first #(= (name attr) (attr-name (.attribute dataset (int %)))) (range (.numAttributes dataset)))))

(defn instance-index-attr
  "Returns the index of an attribute in the attributes definition of an
   instance or dataset"
  [^Instance instance attr]
  (if (number? attr)
    attr
    (find-first #(= (name attr) (attr-name (.attribute instance (int %)))) (range (.numAttributes instance)))))

;; Construction of individual data and datasets

(defn- double-or-nan [x]
  (if (nil? x) Double/NaN (double x)))

(defn make-instance
  "Creates a new dataset instance from a vector"
  ([dataset vector]
     (make-instance dataset 1 vector))
  ([dataset weight vector]
     (let [^Instance inst (new Instance
                     (count vector))]
       (do (.setDataset inst dataset)
           (loop [vs vector
                  c 0]
             (if (empty? vs)
               (doto inst (.setWeight (double weight)))
               (do
                 (if (or (keyword? (first vs)) (string? (first vs)))
                   ;; this is a nominal entry in keyword or string form
                   (.setValue inst (int c) (name (first vs)))
                   (if (sequential? (first vs))
                     ;; this is a map of labels
                     (let [k (name (nth (first vs) 0))
                           val (nth (first vs) 1)
                           ik  (int (instance-index-attr inst k))]
                       (if (or (keyword? val) (string? val))
                         ;; this is a nominal entry in keyword or string form
                         (.setValue inst ik ^String (name val))
                         (.setValue inst ik (double-or-nan val))))
                     ;; A double value for the entry
                     (.setValue inst (int c) (double-or-nan (first vs)))))
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
  [^Instances dataset]
  (.relationName dataset))

(defn dataset-set-name
  "Sets the dataset's name"
  [^Instances dataset ^String new-name]
  (doto dataset (.setRelationName new-name)))

(defn attribute-labels-indexes
  "Returns map of the labels (possible values) for the given nominal attribute as the keys
   with the values being the attributes index. "
  [^Attribute attr]
   (let [values (enumeration-seq (.enumerateValues attr))]
    (if (empty? values)
      :not-nominal
      (reduce (fn [m ^String val]
                (assoc m (keyword val) (.indexOfValue attr val)))
              {}
              values))))

(defn attribute-labels
  "Returns the labels (possible values) for the given nominal attribute as keywords."
  [^Attribute attr]
  (set (map keyword (enumeration-seq (.enumerateValues attr)))))

(defn attribute-labels-as-strings
  "Returns the labels (possible values) for the given nominal attribute as strings."
  [^Attribute attr]
  (set (enumeration-seq (.enumerateValues attr))))

(defn dataset-labels-at [dataset-or-instance index-or-name]
  "Returns the lables (possible values) for a nominal attribute at the provided position"
  (attribute-labels-indexes
   (attribute-at dataset-or-instance index-or-name)))

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

(defn headers-only
  "Returns a new weka dataset (Instances) with the same headers as the given one"
  [^Instances ds]
  (Instances. ds 0))

(defn dataset-class-index
  "Returns the index of the class attribute for this dataset"
  [^Instances dataset]
  (.classIndex dataset))

(defn dataset-class-name
  "Returns the name of the class attribute in keyword form.  Returns nil if not set."
  [^Instances dataset]
  (when (> (dataset-class-index dataset) -1)
    (keyword-name (.classAttribute dataset))))

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
  "Sets the value (label) of the class attribute for this instance"
  [^Instance instance ^String val]
  (doto instance (.setClassValue val)))

(defn instance-get-class
  "Get the index of the class attribute for this instance"
   [^Instance instance]
  (.classValue instance))

(defn instance-value-at
  "Returns the value of an instance attribute. A string, not a keyword is returned."
  [^Instance instance pos]
  (let [pos (int pos)
        attr (.attribute instance pos)
        val (.value instance pos)]
    (if (Double/isNaN val)
      nil
      (if (.isNominal attr) ; This ignores the fact that weka can have date and other attribute types...
        (.stringValue instance pos)
        val))))

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
    (seq (enumeration-seq (.enumerateInstances ^Instances dataset)))))

(defn dataset-as-maps
  "Returns a lazy sequence of the dataset represetned as maps.
This fn is preferale to mapping over a seq yourself with instance-to-map
becuase it avoids redundant string interning of the attribute names."
  [dataset]
  (let [attrs (attribute-names dataset)] ; we only want to intern the attribute names once!
    (for [instance (map instance-to-list (dataset-seq dataset))]
      (zipmap attrs instance))))

(defn dataset-as-lists
  "Returns a lazy sequence of the dataset represented as lists.  The values
   are the actual values (i.e. the string values) and not weka's internal
   double representation or clj-ml's keyword representation."
  [dataset]
  (map instance-to-list (dataset-seq dataset)))

(defn dataset-as-vecs
  "Returns a lazy sequence of the dataset represented as lists.  The values
   are the actual values (i.e. the string values) and not weka's internal
   double representation or clj-ml's keyword representation."
  [dataset]
  (map instance-to-vector (dataset-seq dataset)))

(defn dataset-set-class
  "Sets the index of the attribute of the dataset that is the class of the dataset"
  [^Instances dataset index-or-name]
  (doto dataset (.setClassIndex ^int (dataset-index-attr dataset index-or-name))))

(defn dataset-remove-class
  "Removes the class attribute from the dataset"
  [^Instances dataset]
  (doto dataset (.setClassIndex -1)))

(defn dataset-count
  "Returns the number of elements in a dataset"
  [^Instances dataset]
  (.numInstances dataset))

(defn dataset-add
  "Adds a new instance to a dataset. A clojure vector, map, or an Instance
   can be passed as arguments"
  ([dataset vector]
     (dataset-add dataset 1 vector))
  ([^Instances dataset weight vector]
     (doto dataset
       (.add ^Instance (if (is-instance? vector)
                         vector
                         (make-instance dataset weight vector))))))

(defn dataset-extract-at
  "Removes and returns the instance at a certain position from the dataset"
  [^Instances dataset pos]
  (let [inst (.instance dataset (int pos))]
    (do
      (.delete dataset (int pos))
      inst)))

(defn dataset-at
  "Returns the instance at a certain position from the dataset"
  [^Instances dataset pos]
  (.instance dataset (int pos)))

(defn dataset-pop
  "Removes and returns the first instance in the dataset"
  [dataset]
  (dataset-extract-at dataset 0))

(defn dataset-replace-attribute!
  "Replaces the specified attribute with the given one. (The attribute should be a weka.core.Attribute)
This function only modifies the format of the dataset and does not deal with any instances.
The intention is for this to be used on data-formats and not on datasets with data."
  [^Instances dataset attr-name ^Attribute new-attr]
  (let [attr-pos (dataset-index-attr dataset attr-name)]
    (doto dataset
      (.deleteAttributeAt (int attr-pos))
      (.insertAttributeAt new-attr (int attr-pos)))))
