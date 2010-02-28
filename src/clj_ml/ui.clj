;;
;; User interface utilities
;; @author Antonio Garrote
;;

(ns clj-ml.ui
  (:use (incanter core stats charts)
        (clj-ml data utils)))

(defmulti display-object
  "Displays some kind of clj-ml object"
  (fn [kind chart data opts] [kind chart]))

(defmethod display-object [:dataset :boxplot]
  ([kind chart dataset-opts display-opts]
     (let [dataset (get dataset-opts :dataset)
           dataseq (dataset-seq dataset)
           cols    (get dataset-opts :cols)
           cols-names (dataset-attributes-definition dataset)
           vals-map (reduce (fn [acum col]
                              (let [name (key-to-str (nth cols-names col))
                                    vals (map #(nth (instance-to-vector %1) col) dataseq)]
                                (conj acum {name vals})))
                            {}
                            cols)
           title (or (get display-opts :title) (str "Dataset '" (dataset-name dataset) "' Box Plot"))
           legend (or (get display-opts :legend) true)]
       (loop [plot nil
              ks (keys vals-map)]
         (if (empty? ks)
           (do
             (view plot)
             plot)
           (let [this-val (get vals-map (first ks))
                 the-plot (if (nil? plot)
                            (box-plot this-val :title title)
                            (do (add-box-plot plot this-val)
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
           cols-names (dataset-attributes-definition dataset)
           group-vals (dataset-values-at dataset group-by)
           acum-map (reduce (fn [acum group-val]
                              (conj acum {(first group-val) (reduce (fn [acum x] (conj acum {x []}))
                                                            {}
                                                            cols)}))
                            {}
                            group-vals)
           folded-points (reduce (fn [acum instance]
                                   (let [inst (instance-to-vector instance)
                                         val-0 (nth inst col-0)
                                         val-1 (nth inst col-1)
                                         class (nth inst group-by)]
                                     (merge-with
                                      (fn [a b] {col-0 (conj (get a col-0)
                                                             (get b col-0))
                                                 col-1 (conj (get a col-1)
                                                             (get b col-1))})
                                      acum
                                      {class {col-0 val-0 col-1 val-1}})
                                     ))
                                 acum-map
                                 dataseq)
           title (or (get display-opts :title) (str "Dataset '" (dataset-name dataset) "' Scatter Plot ("
                                                    (key-to-str (nth cols-names col-0)) " vs "
                                                    (key-to-str (nth cols-names col-1)) ")"))
           legend (or (get display-opts :legend) true)]
       (loop [plot nil
              ks (keys folded-points)]
         (if (empty? ks)
           (do
             (view plot)
             plot)
           (let [this-vals  (get folded-points (first ks))
                 this-val-0 (get this-vals col-0)
                 this-val-1 (get this-vals col-1)
                 the-plot (if (nil? plot)
                            (scatter-plot this-val-0 this-val-1
                                          :title title
                                           :x-label (key-to-str (nth cols-names col-0))
                                           :y-label (key-to-str (nth cols-names col-1)))
                             (do (add-points plot this-val-0 this-val-1)
                                 plot))]
              (recur the-plot (rest ks))))))))


;; Things to load to test this from slime

;(defn load-test-from-slime []
;  (do
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
