;;
;; Common utilities and functions
;; @author Antonio Garrote
;;

(ns #^{:author "Antonio Garrote <antoniogarrote@gmail.com>"
       :skip-wiki true}
  clj-ml.utils
  (:import (java.io ObjectOutputStream ByteArrayOutputStream
                    ByteArrayInputStream ObjectInputStream
                    FileOutputStream FileInputStream)
           (java.security
            NoSuchAlgorithmException
            MessageDigest)))


(defn key-to-str
  "transforms a keyword into a string"
  ([k]
     (if (= (class k) String)
       k
       (let [sk (str k)]
         (.substring sk 1)))))

(defn first-or-default
  "Returns the first element in the collection or the default value"
  ([col default]
     (if (empty? col)
       default
       (first col))))

(defn into-fast-vector
  "Similar to into-array but returns a weka.core.FastVector"
  [coll]
  (let [fv (weka.core.FastVector.)]
    (doseq [item coll]
      (.addElement fv item))
    fv))

(defn update-in-when
  "Similar to update-in, but returns m unmodified if any levels do
  not exist"
  ([m [k & ks] f & args]
   (if (contains? m k)
     (if ks
       (assoc m k (apply update-in-when (get m k) ks f args))
       (assoc m k (apply f (get m k) args)))
     m)))

;; trying metrics

(defn try-metric [f]
  (try (f)
       (catch Exception ex {:nan (.getMessage ex)})))

(defn try-multiple-values-metric [class-values f]
  (loop [acum {}
         ks (keys class-values)]
    (if (empty? ks)
      acum
      (let [index (get class-values (first ks))
            val (f index)]
        (recur (conj acum {(first ks) val})
               (rest ks))))))


(defn md5-sum
  "Compute the hex MD5 sum of a string."
  [#^String str]
  (let [alg (doto (MessageDigest/getInstance "MD5")
              (.reset)
              (.update (.getBytes str)))]
    (try
      (.toString (new BigInteger 1 (.digest alg)) 16)
      (catch NoSuchAlgorithmException e
        (throw (new RuntimeException e))))))

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

;; Serializing classifiers

(defn serialize
  "Writes an object to memory"
  ([obj]
     (let [bs (new ByteArrayOutputStream)
           os (new ObjectOutputStream bs)]
       (.writeObject os obj)
       (.close os)
       (.toByteArray bs))))

(defn deserialize
  "Reads an object from memory"
  ([bytes]
     (let [bs (new ByteArrayInputStream bytes)
           is (new ObjectInputStream bs)
           obj (.readObject is)]
       (.close is)
       obj)))

(defn serialize-to-file
  "Writes an object to a file"
  ([obj path]
     (let [fs (new FileOutputStream path)
           os (new ObjectOutputStream fs)]
       (.writeObject os obj)
       (.close os))
     path))

(defn deserialize-from-file
  "Reads an object from a file"
  ([path]
     (let [fs (new FileInputStream path)
           is (new ObjectInputStream fs)
           obj (.readObject is)]
       (.close is)
       obj)))
