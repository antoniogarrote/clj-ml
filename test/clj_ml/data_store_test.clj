(ns clj-ml.data-store-test
  (:use [clj-ml.data-store] :reload-all)
  (:use [clojure.test]))

(deftest make-instance-num
  (is (= (keywords-to-strings
          [1 :hola {:a [:b {:d "hola"}]}])
          '(1 "hola" {"a" ("b" {"d" "hola"})}))))
