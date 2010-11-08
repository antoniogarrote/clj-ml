(ns clj-ml.utils-test
  (:use [clj-ml utils] :reload-all)
  (:use [clojure.test]))

(deftest test-into-fast-vecotor
  (is (= ["a" "B" "c"]
           (vec (.toArray (into-fast-vector ["a" "B" "c"]))))))
