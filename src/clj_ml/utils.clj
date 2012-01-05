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

;; taken from clojure.contrib.seq
(defn find-first
  "Returns the first item of coll for which (pred item) returns logical true.
  Consumes sequences up to the first match, will consume the entire sequence
  and return nil if no match is found."
  [pred coll]
  (first (filter pred coll)))

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

(defn map-fast-vec [^weka.core.FastVector fast-vector f]
  (->> (.elements fast-vector)
       enumeration-seq
       (map f)
       into-fast-vector))

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
