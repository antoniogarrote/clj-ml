(ns clj-ml.attribute-selection-test
  (:use [clj-ml attribute-selection data] :reload-all)
  (:use clojure.test midje.sweet))

(deftest make-greedy-options
  (let [options (make-obj-options :greedy {:generate-rankings true :threshold 0.2
                                           :num-attributes 4 :direction :backward})]
    (are [index expected-flag] (is (= (get options index) expected-flag))
         0 "-R"
         1 "-T"
         2 "0.2"
         3 "-N"
         4 "4"
         5 "-B")))

(deftest select-attributes-test
  (let [ds (make-dataset :test [:a :b {:c [:g :m]}]
                         [ [1 2 :g]
                           [2 3 :m]
                           [4 5 :g]])
        attrs (select-attributes ds :search (greedy) :evaluator (cfs-subset-eval))]
    (facts
      attrs => [:a :c]
      (-> attrs meta :selector class) => #(isa? weka.attributeSelection.AttributeSelection %))))

