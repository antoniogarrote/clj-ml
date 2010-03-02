(ns clj-ml.distance-functions-test
  (:use [clj-ml distance-functions] :reload-all)
  (:use [clojure.test]))

(deftest make-distance-function-euclidean
  (let [dist (clj-ml.distance-functions/make-distance-function :euclidean {:attributes [0 1 2 3]})
        options (.getOptions dist)]
    (is (= (aget options 0)
           "-R"))
    (is (= (aget options 1)
           "1,2,3,4"))))

(deftest make-distance-function-manhattan
  (let [dist (clj-ml.distance-functions/make-distance-function :manhattan {:attributes [0 1 2 3]})
        options (.getOptions dist)]
    (is (= (aget options 0)
           "-R"))
    (is (= (aget options 1)
           "1,2,3,4"))))

(deftest make-distance-function-chebyshev
  (let [dist (clj-ml.distance-functions/make-distance-function :chebyshev {:attributes [0 1 2 3]})
        options (.getOptions dist)]
    (is (= (aget options 0)
           "-R"))
    (is (= (aget options 1)
           "1,2,3,4"))))


