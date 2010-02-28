;;
;; Manipulation of datasets and instances
;; @author Antonio Garrote
;;

(ns clj-ml.data
  (:use [clj-ml utils])
  (:import (weka.core Instance Instances FastVector Attribute)
           (cljml ClojureInstances)))


;; Construction of individual data and datasets

(defn attribute-name-at- [dataset-or-instance pos]
  (let [class-attr (.attribute dataset-or-instance pos)]
    (.name class-attr)))

(defn- index-attr [dataset-or-instance attr]
  (let [max (.numAttributes dataset-or-instance)
        attrs (key-to-str attr)]
    (loop [c 0]
      (if (= c max)
        (throw (.Exception (str "Attribute " attrs " not found")))
        (if (= attrs (attribute-name-at- dataset-or-instance c))
          c
          (recur (+ c 1 )))))))

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
               (do
                 (.setWeight inst (double weight))
                 inst)
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
  "Creates a new empty dataset. By default the class is set to be the last attribute."
  ([name attributes capacity-or-values]
     (make-dataset name attributes 1 capacity-or-values))
  ([name attributes weight capacity-or-values]
     (let [ds (if (sequential? capacity-or-values)
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
       ;; by default the class is the last attribute in the dataset
       ;; (.setClassIndex ds (- (.numAttributes ds) 1))
       ds)))

;; dataset information

(defn dataset-class-values [dataset]
  "Returns the possible values for the class attribute"
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

(defn dataset-attributes-definition [dataset]
  "Returns the definition of the attributes of this dataset"
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

;; manipulation of instances

(defn instance-set-class [instance pos]
  "Sets the index of the class attribute for this instance"
  (do (.setClassValue instance pos)
      instance))

(defn instance-get-class [instance]
  "Get the index of the class attribute for this instance"
  (.classValue instance))

(defn instance-value-at [instance pos]
  "Returns the value of an instance attribute"
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

(defn dataset-seq [dataset]
  "Builds a new clojure sequence from this dataset"
  (if (= (class dataset)
         ClojureInstances)
    (seq dataset)
    (seq (new ClojureInstances dataset))))

(defn dataset-set-class [dataset pos]
  "Sets the index of the attribute of the dataset that is the class of the dataset"
  (do (.setClassIndex dataset pos)
      dataset))

(defn dataset-remove-class [dataset]
  "Removes the class attribute from the dataset"
  (do
    (.setClassIndex dataset -1)
    dataset))

(defn dataset-count [dataset]
  "Returns the number of elements in a dataset"
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
