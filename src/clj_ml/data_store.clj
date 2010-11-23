;;
;; Distance functions
;; @author Antonio Garrote
;;

(ns #^{:author "Antonio Garrote <antoniogarrote@gmail.com>"}
  clj-ml.data-store
  "Functions for storing and retrieving data sets from a persistence store, like a data
   base system.
   Currently MongoDB is the only store supported."
  (:use [clj-ml utils data])
  (:import (com.mongodb Mongo DB BasicDBObject DBCollection DBCursor)))

(defn keywords-to-strings [format]
  "Recursively transforms all keywords into strings"
  (if (keyword? format)
    (name format)
    (if (map? format)
      (loop [acum {}
             ks (keys format)]
        (if (empty? ks)
          acum
          (recur (conj {(name (first ks))
                        (keywords-to-strings (get format (first ks)))}
                       acum)
                 (rest ks))))
      (if (sequential? format)
        (map #(keywords-to-strings %1) format)
        format))))

(defmulti make-data-store-connection
  "Connects to a data store.

   - The first parameter is the kind of data store to connect to.
   - The second parameter is a map with options for the connection
     to that kind of data store.
"
  (fn [kind params] kind))

(defmethod make-data-store-connection :mongodb
  ([kind params]
     (let [_foo1 (println (:host params))
           _foo2 (println (:port params))]
       (new Mongo (:host params) (:port params)))
     ))


(defmulti data-store-connection-db
  "Returns a DB from the stablished connection"
  (fn [kind connection name & params] kind))

(defmethod data-store-connection-db :mongodb
  ([kind connection name & params]
     (.getDB connection name)))

;; High level API

(def *clj-ml-datasets* "-clj-ml-datasets")
(def *clj-ml-format-suffix* "-clj-ml-schema")
(def *clj-ml-instances-suffix* "-clj-ml-instances")

(defmulti data-store-save-dataset
  "Persists a whole dataset in the data store"
  (fn [kind database dataset & options] kind))

(defmethod data-store-save-dataset :mongodb
  ([kind database dataset & options]
     (let [format (dataset-format dataset)
           name (md5-sum (dataset-name dataset))
           datasets-collection (.getCollection database *clj-ml-datasets*)
           schema-collection (.getCollection database (str name *clj-ml-format-suffix*))
           data-collection-tmp (.getCollection database (str name *clj-ml-instances-suffix*))
           format-to-insert (new BasicDBObject {"format" (keywords-to-strings format)})]
       (.remove datasets-collection (new BasicDBObject {"id" name}))
       (.insert datasets-collection (new BasicDBObject {"id" name}))
       (.remove schema-collection format-to-insert)
       (.insert schema-collection format-to-insert)
       (when (not (nil? data-collection-tmp))
         (.drop data-collection-tmp))
       (let [data-collection (.getCollection database (str name *clj-ml-instances-suffix*))]
         (for [i (dataset-seq dataset)]
           (.insert data-collection (new BasicDBObject (keywords-to-strings {"instance" (instance-to-vector i)}))))))))

(declare mongo-persisted-instance-to-map)

(defn- mongo-persisted-instance-to-vector
  "Transforms an instance persisted in a mongodb database back to a vector"
  ([inst] (mongo-persisted-instance-to-vector inst false))
  ([inst use-keys?]
     (loop [vals (.toMap inst)
            acum []]
       (if (empty? vals)
         acum
         (recur (rest vals)
                (conj acum (let [tmp-val (.getValue (first vals))]
                             (if (= (class tmp-val) com.mongodb.BasicDBObject)
                               (mongo-persisted-instance-to-map tmp-val use-keys?)
                               (if (string? tmp-val)
                                 (if use-keys? (keyword tmp-val) tmp-val)
                                 tmp-val)))))))))

(defn- mongo-persisted-instance-to-map
  "Transforms an instance persisted in a mongodb database back to a vector"
  ([inst] (mongo-persisted-instance-to-map inst false))
  ([inst use-keys?]
     (loop [mp (.toMap inst)
            vals (keys mp)
            acum {}]
       (if (empty? vals)
         acum
         (recur mp
                (rest vals)
                (conj acum {(keyword (first vals)) (let [tmp-val (get mp (first vals))]
                                                     (if (= (class tmp-val) com.mongodb.BasicDBObject)
                                                       (mongo-persisted-instance-to-map tmp-val use-keys?)
                                                       (if (= (class tmp-val) com.mongodb.BasicDBList)
                                                         (mongo-persisted-instance-to-vector tmp-val use-keys?)
                                                         tmp-val)))}))))))


(defmulti data-store-load-dataset
  "Load a whole dataset from a data store"
  (fn [kind database database-name & options] kind))

(defmethod data-store-load-dataset :mongodb
  ([kind database database-name & options]
     (let [dsf (str (md5-sum database-name) *clj-ml-format-suffix*)
           col (.getCollection database dsf)
           format (mongo-persisted-instance-to-vector (get (.next (.find col)) "format") true)
           dsi (str (md5-sum database-name) *clj-ml-instances-suffix*)
           coli (.getCollection database dsi)
           cursor (.find coli)
           insts (loop [cont (.hasNext cursor)
                        acum []]
                   (if cont
                     (let [exp (get (. (.next cursor) toMap) "instance")]
                       (recur (.hasNext cursor)
                              (conj acum (mongo-persisted-instance-to-vector exp))))
                     acum))]
       (make-dataset database-name format insts))))

