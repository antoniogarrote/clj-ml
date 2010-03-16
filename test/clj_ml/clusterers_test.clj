(ns clj-ml.clusterers-test
  (:use [clj-ml clusterers data] :reload-all)
  (:use [clojure.test]))

(deftest make-clusterers-options-k-means
  (let [options (make-clusterer-options :k-means {:display-standard-deviation true :replace-missing-values true :preserve-instances-order true
                                                  :number-clusters 3 :random-seed 2 :number-iterations 1})]
    (is (= (aget options 0)
           ""))
    (is (= (aget options 1)
           "-V"))
    (is (= (aget options 2)
           "-M"))
    (is (= (aget options 3)
           "-O"))
    (is (= (aget options 4)
           "-N"))
    (is (= (aget options 5)
           "3"))
    (is (= (aget options 6)
           "-S"))
    (is (= (aget options 7)
           "2"))
    (is (= (aget options 8)
           "-I"))
    (is (= (aget options 9)
           "1"))))

(deftest make-clusterers-options-expectation-maximization
  (let [options (make-clusterer-options :expectation-maximization {:number-clusters 3 :maximum-iterations 10 :minimum-standard-deviation 0.001 :random-seed 30})]
    (is (= (aget options 0)
           ""))
    (is (= (aget options 1)
           "-N"))
    (is (= (aget options 2)
           "3"))
    (is (= (aget options 3)
           "-I"))
    (is (= (aget options 4)
           "10"))
    (is (= (aget options 5)
           "-M"))
    (is (= (aget options 6)
           "0.0010"))
    (is (= (aget options 7)
           "-S"))
    (is (= (aget options 8)
           "30"))))


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
