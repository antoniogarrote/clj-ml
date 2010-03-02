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


(deftest make-and-build-classifier
  (let [ds (make-dataset :test [:a :b] [[1 2] [3 4]])
        c  (make-clusterer :k-means)]
    (clusterer-build c ds)
    (is (= weka.clusterers.SimpleKMeans (class c)))))
