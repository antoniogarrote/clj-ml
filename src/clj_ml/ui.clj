;;
;; User interface utilities
;; @author Antonio Garrote
;;

(ns clj-ml.ui
  (:use (clj-ml data utils clusterers)
        (incanter core stats charts))
  (:import (weka.clusterers ClusterEvaluation SimpleKMeans)))


(defn visualize-plot [plot]
  "Prepare a plot to be displayed"
  (do (clear-background plot)
      (view plot)
      plot))

(defmulti display-object
  "Displays some kind of clj-ml object"
  (fn [kind chart data opts] [kind chart]))

(defmethod display-object [:dataset :boxplot]
  ([kind chart dataset-opts display-opts]
     (let [dataset (get dataset-opts :dataset)
           dataseq (dataset-seq dataset)
           cols    (get dataset-opts :cols)
           cols-names (dataset-format dataset)
           vals-map (reduce (fn [acum col]
                              (let [name (key-to-str (nth cols-names col))
                                    vals (map #(nth (instance-to-vector %1) col) dataseq)]
                                (conj acum {name vals})))
                            {}
                            cols)
           title (or (get display-opts :title) (str "Dataset '" (dataset-name dataset) "' Box Plot"))
           legend (if (nil? (get display-opts :legend))  true (get display-opts :legend))
           should-display (get display-opts :visualize)]
       (loop [plot nil
              ks (keys vals-map)]
         (if (empty? ks)
           (if should-display
             (visualize-plot plot)
             plot)
           (let [this-val (get vals-map (first ks))
                 the-plot (if (nil? plot)
                            (box-plot this-val :title title :legend legend :series-label (key-to-str (first ks)))
                            (do (add-box-plot plot this-val :series-label (key-to-str (first ks)))
                                plot))]
             (recur the-plot (rest ks))))))))


(defmethod display-object [:dataset :scatter-plot]
  ([kind chart dataset-opts display-opts]
     (let [dataset (get dataset-opts :dataset)
           dataseq (dataset-seq dataset)
           cols    (get dataset-opts :cols)
           col-0 (nth cols 0)
           col-1 (nth cols 1)
           group-by (get dataset-opts :group-by)
           cols-names (dataset-format dataset)
           group-vals (if (nil? group-by) {:no-group-by :no-class} (dataset-values-at dataset group-by))
           acum-map (reduce (fn [acum group-val]
                              (conj acum {(first group-val)
                                          (reduce (fn [acum x] (conj acum {x []}))
                                                  {}
                                                  cols)}))
                            {}
                            group-vals)
           folded-points (reduce (fn [acum instance]
                                   (let [inst (instance-to-vector instance)
                                         val-0 (nth inst col-0)
                                         val-1 (nth inst col-1)
                                         class (if (nil? group-by)
                                                 :no-group-by
                                                 (nth inst group-by))]
                                     (merge-with
                                      (fn [a b] {col-0 (conj (get a col-0)
                                                             (get b col-0))
                                                 col-1 (conj (get a col-1)
                                                             (get b col-1))})
                                      acum
                                      {class {col-0 val-0 col-1 val-1}})))
                                 acum-map
                                 dataseq)
           title (or (get display-opts :title) (str "Dataset '" (dataset-name dataset) "' Scatter Plot ("
                                                    (key-to-str (nth cols-names col-0)) " vs "
                                                    (key-to-str (nth cols-names col-1)) ")"))
           legend (if (nil? (get display-opts :legend))  true (get display-opts :legend))
           should-display (get display-opts :visualize)]
       (loop [plot nil
              ks (keys folded-points)]
         (if (empty? ks)
           (if should-display
             (visualize-plot plot)
             plot)
           (let [this-vals  (get folded-points (first ks))
                 this-val-0 (get this-vals col-0)
                 this-val-1 (get this-vals col-1)
                 the-plot (if (nil? plot)
                            (scatter-plot this-val-0 this-val-1
                                          :title title
                                           :x-label (key-to-str (nth cols-names col-0))
                                           :y-label (key-to-str (nth cols-names col-1))
                                           :series-label (key-to-str (first ks))
                                           :legend legend)
                             (do (add-points plot this-val-0 this-val-1 :series-label (key-to-str (first ks)))
                                 plot))]
              (recur the-plot (rest ks))))))))


;; visualization of different objects

(defn dataset-display-numeric-attributes [dataset attributes & visualization-options]
  "Displays the provided attributes into a box plot"
  (let [attr (map #(if (keyword? %1) (index-attr dataset %1) %1) attributes)
        options-pre (first-or-default visualization-options {})
        options (if (nil? (:visualize options-pre)) (conj options-pre {:visualize true}) options-pre)]
    (display-object :dataset :boxplot {:dataset dataset :cols attr} options)))

(defn dataset-display-class-for-attributes [dataset attribute-x attribute-y & visualization-options]
  "Displays how a pair of attributes are distributed for each class"
  (let [attr-x (if (keyword? attribute-x) (index-attr dataset attribute-x) attribute-x)
        attr-y (if (keyword? attribute-y) (index-attr dataset attribute-y) attribute-y)
        options-pre (first-or-default visualization-options {})
        opts (if (nil? (:visualize options-pre)) (conj options-pre {:visualize true}) options-pre)
        class-index (dataset-get-class dataset)]
    (display-object :dataset :scatter-plot {:dataset dataset :cols [attr-x attr-y] :group-by class-index} opts)))

(defn dataset-display-attributes [dataset attribute-x attribute-y & visualization-options]
  "Displays the distribution of a set of attributes for a dataset"
    (let [attr-x (if (keyword? attribute-x) (index-attr dataset attribute-x) attribute-x)
        attr-y (if (keyword? attribute-y) (index-attr dataset attribute-y) attribute-y)
        options-pre (first-or-default visualization-options {})
        opts (if (nil? (:visualize options-pre)) (conj options-pre {:visualize true}) options-pre)
        class-index (dataset-get-class dataset)]
    (display-object :dataset :scatter-plot {:dataset dataset :cols [attr-x attr-y]} opts)))


;; visualization

(defmulti clusterer-display-for-attributes
  (fn [clusterer dataset attribute-x attribute-y] (class clusterer)))

(defmethod clusterer-display-for-attributes SimpleKMeans
  ([clusterer dataset attribute-x attribute-y & visualization-options]
     (let [attr-x (if (keyword? attribute-x) (instance-index-attr dataset attribute-x) attribute-x)
           attr-y (if (keyword? attribute-y) (instance-index-attr dataset attribute-y) attribute-y)
           opts (first-or-default visualization-options {})
           display? (if (= (get visualization-options :visualize) false)
                      false
                      true)
           true-opts (conj opts {:visualize false})
           plot (dataset-display-class-for-attributes dataset attribute-x attribute-y true-opts)
           info (clusterer-info clusterer)
           centroids (:centroids info)]
       (do
         (loop [ks (keys centroids)]
           (if (empty? ks)
             (if display?
               (visualize-plot plot)
               plot)
             (let [k (first ks)
                   centroid (get centroids k)
                   val-x (instance-value-at centroid attr-x)
                   val-y (instance-value-at centroid attr-y)]
               (add-pointer plot val-x val-y :text (str "centroid " k " (" (float val-x) "," (float val-y) ")"))
               (recur (rest ks)))))))))



;; Things to load to test this from slime

;(defn load-test-from-slime []
;  (do
;    (add-classpath "file:///Users/antonio.garrote/Development/old/clj-ml/lib/joda-time-1.6.jar")
;    (add-classpath "file:///Users/antonio.garrote/Development/old/clj-ml/lib/opencsv-2.0.1.jar")
;    (add-classpath "file:///Users/antonio.garrote/Development/old/clj-ml/classes/")
;    (add-classpath "file:///Applications/weka-3-6-2/weka.jar")
;    (add-classpath "file:///Users/antonio.garrote/Development/old/clj-ml/src/")
;    (add-classpath "file:///Users/antonio.garrote/Development/old/clj-ml/lib/incanter-charts-1.0-master-SNAPSHOT.jar")
;    (add-classpath "file:///Users/antonio.garrote/Development/old/clj-ml/lib/incanter-core-1.0-master-SNAPSHOT.jar")
;    (add-classpath "file:///Users/antonio.garrote/Development/old/clj-ml/lib/incanter-io-1.0-master-SNAPSHOT.jar")
;    (add-classpath "file:///Users/antonio.garrote/Development/old/clj-ml/lib/incanter-processing-1.0-master-SNAPSHOT.jar")
;    (add-classpath "file:///Users/antonio.garrote/Development/old/clj-ml/lib/incanter-chrono-1.0-master-SNAPSHOT.jar")
;    (add-classpath "file:///Users/antonio.garrote/Development/old/clj-ml/lib/incanter-full-1.0-master-SNAPSHOT.jar")
;    (add-classpath "file:///Users/antonio.garrote/Development/old/clj-ml/lib/incanter-mongodb-1.0-master-SNAPSHOT.jar")
;    (add-classpath "file:///Users/antonio.garrote/Development/old/clj-ml/lib/jfreechart-1.0.13.jar")
;    (add-classpath "file:///Users/antonio.garrote/Development/old/clj-ml/lib/parallelcolt-0.7.2.jar")
;    (add-classpath "file:///Users/antonio.garrote/Development/old/clj-ml/lib/arpack-combo-0.1.jar")
;    (add-classpath "file:///Users/antonio.garrote/Development/old/clj-ml/lib/gnujaxp-1.jar")
;    (add-classpath "file:///Users/antonio.garrote/Development/old/clj-ml/lib/clojure-json-1.1-20091229.021828-4.jar")
;    (add-classpath "file:///Users/antonio.garrote/Development/old/clj-ml/lib/clojure-db-object-0.1.1-20091229.021828-2.jar")
;    (add-classpath "file:///Users/antonio.garrote/Development/old/clj-ml/lib/jcommon-1.0.16.jar")
;    (add-classpath "file:///Users/antonio.garrote/Development/old/clj-ml/lib/netlib-java-0.9.1.jar")
;    (add-classpath "file:///Users/antonio.garrote/Development/old/clj-ml/lib/processing-core-1.jar")
;    ))
