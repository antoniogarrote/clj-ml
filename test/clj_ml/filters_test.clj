(ns clj-ml.filters-test
  (:use [clj-ml.filters] :reload-all)
  (:use [clojure.test]))

(deftest make-filter-options-supervised-discretize
  (let [options (make-filter-options :supervised-discretize {:attributes [1 2] :invert true :binary true :better-encoding true :kononenko true :nonexitent true})]
    (is (= (aget options 0)
           "-R"))
    (is (= (aget options 1)
           "2,3"))
    (is (= (aget options 2)
           "-V"))
    (is (= (aget options 3)
           "-D"))
    (is (= (aget options 4)
           "-E"))
    (is (= (aget options 5)
           "-K"))))

(deftest make-filter-options-unsupervised-discretize
  (let [options (make-filter-options :unsupervised-discretize {:attributes [1 2] :binary true :better-encoding true
                                                               :better-encoding true :equal-frequency true :optimize true
                                                               :number-bins 4 :weight-bins 1})]
    (is (= (aget options 0)
           "-R"))
    (is (= (aget options 1)
           "2,3"))
    (is (= (aget options 2)
           "-D"))
    (is (= (aget options 3)
           "-E"))
    (is (= (aget options 4)
           "-F"))
    (is (= (aget options 5)
           "-O"))
    (is (= (aget options 6)
           "-B"))
    (is (= (aget options 7)
           "4"))
    (is (= (aget options 8)
           "-M"))
    (is (= (aget options 9)
           "1"))))

(deftest make-filter-options-supervised-nominal-to-binary
  (let [options (make-filter-options :supervised-nominal-to-binary {:also-binary true :for-each-nominal true})]
    (is (= (aget options 0)
           ""))
    (is (= (aget options 1)
           "-N"))
    (is (= (aget options 2)
           "-A"))))

(deftest make-filter-options-unsupervised-nominal-to-binary
  (let [options (make-filter-options :unsupervised-nominal-to-binary {:attributes [1,2] :also-binary true :for-each-nominal true :invert true})]
    (is (= (aget options 0)
           "-R"))
    (is (= (aget options 1)
           "2,3"))
    (is (= (aget options 2)
           "-V"))
    (is (= (aget options 3)
           "-N"))
    (is (= (aget options 4)
           "-A"))))

(deftest make-filter-discretize-sup
  (let [ds (clj-ml.data/make-dataset :test [:a :b {:c [:g :m]}]
                                     [ [1 2 :g]
                                       [2 3 :m]
                                       [4 5 :g]])
        f (make-filter :supervised-discretize {:dataset ds :attributes [0]})]
    (is (= weka.filters.supervised.attribute.Discretize
           (class f)))))

(deftest make-filter-discretize-unsup
  (let [ds (clj-ml.data/make-dataset :test [:a :b {:c [:g :m]}]
                                     [ [1 2 :g]
                                       [2 3 :m]
                                       [4 5 :g]])
        f (make-filter :unsupervised-discretize {:dataset ds :attributes [0]})]
    (is (= weka.filters.unsupervised.attribute.Discretize
           (class f)))))

(deftest make-filter-nominal-to-binary-sup
  (let [ds (clj-ml.data/make-dataset :test [:a :b {:c [:g :m]}]
                                     [ [1 2 :g]
                                       [2 3 :m]
                                       [4 5 :g]])
        f (make-filter :supervised-nominal-to-binary {:dataset ds})]
    (is (= weka.filters.supervised.attribute.NominalToBinary
           (class f)))))

(deftest make-filter-nominal-to-binary-unsup
  (let [ds (clj-ml.data/make-dataset :test [:a :b {:c [:g :m]}]
                                     [ [1 2 :g]
                                       [2 3 :m]
                                       [4 5 :g]])
        f (make-filter :unsupervised-nominal-to-binary {:dataset ds :attributes [2]})]
    (is (= weka.filters.unsupervised.attribute.NominalToBinary
           (class f)))))
