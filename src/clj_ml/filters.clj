;;
;; Data processing of data with different filtering algorithms
;; @author Antonio Garrote
;;

(ns #^{:author "Antonio Garrote <antoniogarrote@gmail.com>"}
  clj-ml.filters
  "This namespace defines a set of functions that can be applied to data sets to modify the
   dataset in some way: transforming nominal attributes into binary attributes, removing
   attributes etc.

   A sample use of the API is shown below:

     ;; *ds* is the dataset where the first attribute is to be removed
     (def *filter* (make-filter :remove-attributes {:dataset-format *ds* :attributes [0]}))

     ;; We apply the filter to the original data set and obtain the new one
     (def *filtered-ds* (filter-apply *filter* *ds*))


   The previous sample of code could be rewritten with the make-apply-filter function:

     ;; There is no necessity of passing the :dataset-format option, *ds* format is used
     ;; automatically
     (def *filtered-ds* (make-apply-filter :remove-attributes {:attributes [0]} *ds*))"
  (:use [clj-ml data utils]
        [clojure.contrib [def :only [defvar]]])
  (:require [clojure.contrib [string :as str]])
  (:import (weka.filters Filter)))


;; Options for the filters

(defmulti  #^{:skip-wiki true}
  make-filter-options
  "Creates the right parameters for a filter. Returns a clojure vector."
  (fn [kind map] kind))

(defn- extract-attributes
  "Transforms the :attributes value from m into the appropriate weka flag"
  [m]
  ["-R" (str/join "," (map inc (:attributes m)))])

(defmethod make-filter-options :supervised-discretize
  ([kind m]
     (->> (extract-attributes m)
          (check-options m {:invert "-V"
                            :binary "-D"
                            :better-encoding "-E"
                            :kononenko "-K"}))))

(defmethod make-filter-options :unsupervised-discretize
  ([kind m]
     (->> (extract-attributes m)
          (check-options m {:unset-class "-unset-class-temporarily"
                            :binary "-D"
                            :better-encoding "-E"
                            :equal-frequency "-F"
                            :optimize "-O"})
          (check-option-values m {:number-bins "-B"
                                  :weight-bins "-M"}))))

(defmethod make-filter-options :supervised-nominal-to-binary
  ([kind m]
     (check-options m {:also-binary "-N" :for-each-nominal "-A"})))

(defmethod make-filter-options :unsupervised-nominal-to-binary
  ([kind m]
     (->> (extract-attributes m)
          (check-options m {:invert "-V"
                            :also-binary "-N"
                            :for-each-nominal "-A"}))))

(defmethod make-filter-options :numeric-to-nominal
  ([kind m]
     (->> (extract-attributes m) (check-options m {:invert "-V"}))))

(defmethod make-filter-options :remove-attributes
  ([kind m]
     (->> (extract-attributes m)
          (check-options m {:invert "-V"}))))

(defmethod make-filter-options :remove-useless-attributes
  ([kind m]
     (check-option-values m {:max-variance "-M"})))

(defmethod make-filter-options :select-append-attributes
  ([kind m]
     (->> (extract-attributes m)
          (check-options m {:invert "-V"}))))

(defmethod make-filter-options :project-attributes
  ([kind options]
     (let [opts (if (nil? (:invert options))
                  (conj options {:invert true})
                  (dissoc options :invert))]
       (make-filter-options :remove-attributes opts))))


;; Creation of filters

(defvar filter-aliases
  {:supervised-discretize weka.filters.supervised.attribute.Discretize
   :unsupervised-discretize weka.filters.unsupervised.attribute.Discretize
   :supervised-nominal-to-binary weka.filters.supervised.attribute.NominalToBinary
   :unsupervised-nominal-to-binary weka.filters.unsupervised.attribute.NominalToBinary
   :numeric-to-nominal weka.filters.unsupervised.attribute.NumericToNominal
   :remove-attributes weka.filters.unsupervised.attribute.Remove
   :remove-useless-attributes weka.filters.unsupervised.attribute.RemoveUseless
   :select-append-attributes weka.filters.unsupervised.attribute.Copy
   :project-attributes weka.filters.unsupervised.attribute.Remove}
  "Mapping of cjl-ml keywords to actual Weka classes")


(defn make-filter
  "Creates a filter for the provided attributes format. The first argument must be a symbol
   identifying the kind of filter to generate.
   Currently the following filters are supported:

     - :supervised-discretize
     - :unsupervised-discretize
     - :supervised-nominal-to-binary
     - :unsupervised-nominal-to-binary
     - :numeric-to-nominal
     - :remove-attributes
     - :remove-useless-attributes
     - :select-append-attributes
     - :project-attributes

    The second parameter is a map of attributes
    for the filter to be built.

    An example of usage could be:

      (make-filter :remove {:attributes [0 1] :dataset-format dataset})

    Documentation for the different filters:

    * :supervised-discretize

      An instance filter that discretizes a range of numeric attributes
      in the dataset into nominal attributes. Discretization is by Fayyad
      & Irani's MDL method (the default).

      Parameters:

        - :attributes
            Index of the attributes to be discretized, sample value: [0,4,6]
        - :invert
            Invert mathcing sense of the columns, sample value: true
        - :kononenko
            Use Kononenko's MDL criterion, sample value: true

    * :unsupervised-discretize

      Unsupervised version of the discretize filter. Discretization is by simple
      pinning.

      Parameters:

        - :attributes
            Index of the attributes to be discretized, sample value: [0,4,6]
        - :dataset-format
            The dataset where the filter is going to be applied or a
            description of the format of its attributes. Sample value:
            dataset, (dataset-format dataset)
        - :unset-class
            Does not take class attribute into account for the application
            of the filter, sample-value: true
        - :binary
        - :equal-frequency
            Use equal frequency instead of equal width discretization, sample
            value: true
        - :optimize
            Optmize the number of bins using leave-one-out estimate of
            estimated entropy. Ingores the :binary attribute. sample value: true
        - :number-bins
            Defines the number of bins to divide the numeric attributes into
            sample value: 3

    * :supervised-nominal-to-binary

      Converts nominal attributes into binary numeric attributes. An attribute with k values
      is transformed into k binary attributes if the class is nominal.

      Parameters:
        - :dataset-format
            The dataset where the filter is going to be applied or a
            description of the format of its attributes. Sample value:
            dataset, (dataset-format dataset)
        - :also-binary
            Sets if binary attributes are to be coded as nominal ones, sample value: true
        - :for-each-nominal
            For each nominal value one binary attribute is created, not only if the
            values of the nominal attribute are greater than two.

    * :unsupervised-nominal-to-binary

      Unsupervised version of the :nominal-to-binary filter

      Parameters:

        - :attributes
            Index of the attributes to be binarized. Sample value: [1 2 3]
        - :dataset-format
            The dataset where the filter is going to be applied or a
            description of the format of its attributes. Sample value:
            dataset, (dataset-format dataset)
        - :also-binary
            Sets if binary attributes are to be coded as nominal ones, sample value: true
        - :for-each-nominal
            For each nominal value one binary attribute is created, not only if the
            values of the nominal attribute are greater than two., sample value: true

    * :remove-attributes

      Remove some columns from the data set after the provided attributes.

      Parameters:

        - :dataset-format
            The dataset where the filter is going to be applied or a
            description of the format of its attributes. Sample value:
            dataset, (dataset-format dataset)
        - :attributes
            Index of the attributes to remove. Sample value: [1 2 3]

    * :remove-useless-attributes

       Remove attributes that do not vary at all or that vary too much. All constant
       attributes are deleted automatically, along with any that exceed the maximum percentage
       of variance parameter. The maximum variance test is only applied to nominal attributes.

     Parameters:

        - :max-variance
            Maximum variance percentage allowed (default 99).
            Note: percentage, not decimal. e.g. 89 not 0.89
            If you pass in a decimal Weka silently sets it to 0.0.

    * :select-append-attributes

      Append a copy of the selected columns at the end of the dataset.

      Parameters:

        - :dataset-format
            The dataset where the filter is going to be applied or a
            description of the format of its attributes. Sample value:
            dataset, (dataset-format dataset)
        - :attributes
            Index of the attributes to remove. Sample value: [1 2 3]
        - :invert
            Invert the selection of the columns. Sample value: [0 1]

    * :project-attributes

      Project some columns from the provided dataset

      Parameters:

        - :dataset-format
            The dataset where the filter is going to be applied or a
            description of the format of its attributes. Sample value:
            dataset, (dataset-format dataset)
        - :invert
            Invert the selection of columns. Sample value: [0 1]"
  [kind options]
  (doto (.newInstance (kind filter-aliases))
    (.setOptions (into-array String (make-filter-options kind options)))
    (.setInputFormat (:dataset-format options))))

;; Processing the filtering of data

(defn filter-apply
  "Filters an input dataset using the provided filter and generates an output dataset. The
   first argument is a filter and the second parameter the data set where the filter should
   be applied."
  [filter dataset]
  (Filter/useFilter dataset filter))

(defn make-apply-filter
  "Creates a new filter with the provided options and apply it to the provided dataset.
   The :dataset-format attribute for the making of the filter will be setup to the
   dataset passed as an argument if no other value is provided.

   The application of this filter is equivalent a the consequetive application of
   make-filter and apply-filter."
  [kind options dataset]
  (let [opts (if (nil? (:dataset-format options)) (conj options {:dataset-format dataset}))
        filter (make-filter kind opts)]
    (filter-apply filter dataset)))
