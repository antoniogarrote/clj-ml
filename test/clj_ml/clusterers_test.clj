(ns clj-ml.clusterers-test
  (:use [clj-ml clusterers data] :reload-all)
  (:use clojure.test midje.sweet))

(deftest make-clusterers-options-k-means
  (fact
    (let [options (vec (make-clusterer-options :k-means {:display-standard-deviation true :replace-missing-values true :preserve-instances-order true
                                                         :number-clusters 3 :random-seed 2 :number-iterations 1}))]
      options => (just ["" "-V" "-M" "-O" "-N" "3" "-S" "2" "-I" "1"] :in-any-order))))

(deftest make-clusterers-options-expectation-maximization
  (fact
    (let [options (vec (make-clusterer-options :expectation-maximization {:number-clusters 3 :maximum-iterations 10 :minimum-standard-deviation 0.001 :random-seed 30}))]
      options => (just ["" "-N" "3" "-I" "10" "-M" "0.0010" "-S" "30"] :in-any-order))))


(deftest make-and-build-clusterer
  (let [ds (make-dataset :test [:a :b] [[1 2] [3 4]])
        c  (make-clusterer :k-means)]
    (clusterer-build c ds)
    (is (= weka.clusterers.SimpleKMeans (class c)))))

(deftest make-clusterer-with-distance
  (let [c (clj-ml.clusterers/make-clusterer :k-means {:distance-function {:manhattan {:attributes [0 1 2]}}})]
    (is (= weka.core.ManhattanDistance (class (.getDistanceFunction c))))))

(deftest test-make-cobweb
  (let [ds (make-dataset :test [:a :b] [[1 2] [3 4]])
        c (make-clusterer :cobweb)]
       (clusterer-build c ds)
       (is true)))

(deftest test-update-clusterer-cobweb
  (let [ds (make-dataset :test [:a :b] [])
        c (make-clusterer :cobweb)]
       (clusterer-build c ds)
       (clusterer-update c (clj-ml.data/make-instance ds [1 2]))
       (is true)))

(deftest test-update-clusterer-cobweb-many-instances
  (let [ds (make-dataset :test [:a :b] [])
        c (make-clusterer :cobweb)
        to-update (make-dataset :test [:a :b] [[1 2] [3 4]])]
       (clusterer-build c ds)
       (clusterer-update c to-update)
       (is true)))

(deftest test-evaluate-clusterer-cross-validation
  (let [ds (make-dataset :test [:a :b] [[1 2] [3 4] [5 6]])
        c (make-clusterer :expectation-maximization)]
       (clusterer-build c ds)
       (clusterer-evaluate c :cross-validation ds 2)
       (is true)))
