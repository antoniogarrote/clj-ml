;;
;; Utilities for converting clojure hash maps into Weka string options
;; @author Ben Mabey
;;

(ns #^{:author "Ben Mabey <ben@benmabey.com>"
       :skip-wiki true}
  clj-ml.options-utils
  (:use [clj-ml data])
  (:require [clojure.contrib [string :as str]]))

;; Manipulation of array of options

(defn check-option [opts val flag map]
  "Sets an option for a filter"
  (let [val-in-map (get map val)]
    (if (nil? val-in-map)
      opts
      (conj opts flag))))

(defn check-option-value [opts val flag map]
  "Sets an option with value for a filter"
  (let [val-in-map (get map val)]
    (if (nil? val-in-map)
      opts
      (conj  (conj opts flag) (str val-in-map)))))


(defn extract-attributes
  "Transforms the :attributes value from m into the appropriate weka flag"
  ([m] (extract-attributes "-R" m))
  ([flag m] (extract-attributes flag :attributes m))
  ([flag key-name m]
     (if-let [attributes (key-name m)]
       [flag (str/join ","
                       (for [attr attributes]
                         (inc (dataset-index-attr (:dataset-format m) attr))))]
       [])))


; TODO: Raise a helpful exception when the keys don't match up with the provided flags.
(defn check-options
  "Checks the presence of a set of options for a filter"
  ([args-map opts-map] (check-options args-map opts-map []))
  ( [args-map opts-map tmp]
      (loop [rem (keys opts-map)
             acum tmp]
        (if (empty? rem)
          acum
          (let [k (first rem)
                vk (get opts-map k)
                rst (rest rem)]
            (recur rst
                   (check-option acum k vk args-map)))))))

(defn check-option-values
  "Checks the presence of a set of options with value for a filter"
  ([args-map opts-map] (check-option-values args-map opts-map []))
  ([args-map opts-map val]
      (loop [rem (keys opts-map)
             acum val]
        (if (empty? rem)
          acum
          (let [k (first rem)
                vk (get opts-map k)
                rst (rest rem)]
            (recur rst
                   (check-option-value acum k vk args-map)))))))
