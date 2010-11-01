(ns clj-ml.filters-test
  (:use [clj-ml filters data] :reload-all)
  (:use [clojure.test]))

(deftest make-filter-options-supervised-discretize
  (let [options (make-filter-options :supervised-discretize {:attributes [1 2] :invert true :binary true :better-encoding true :kononenko true :nonexitent true})]
    (are [index expected-flag] (is (= (get options index) expected-flag))
         0 "-R"
         1 "2,3"
         2 "-V"
         3 "-D"
         4 "-E"
         5 "-K")))

(deftest make-filter-options-unsupervised-discretize
  (let [options (make-filter-options :unsupervised-discretize {:attributes [1 2] :binary true
                                                               :better-encoding true :equal-frequency true :optimize true
                                                               :number-bins 4 :weight-bins 1})]
    (are [index expected-flag] (is (= (get options index) expected-flag))
         0 "-R"
         1 "2,3"
         2 "-D"
         3 "-E"
         4 "-F"
         5 "-O"
         6 "-B"
         7 "4"
         8 "-M"
         9 "1")))

(deftest make-filter-options-supervised-nominal-to-binary
  (let [options (make-filter-options :supervised-nominal-to-binary {:also-binary true :for-each-nominal true})]
    (are [index expected-flag] (is (= (get options index) expected-flag))
         0 "-N"
         1 "-A")))

(deftest make-filter-options-unsupervised-nominal-to-binary
  (let [options (make-filter-options :unsupervised-nominal-to-binary {:attributes [1,2] :also-binary true :for-each-nominal true :invert true})]
    (are [index expected-flag] (is (= (get options index) expected-flag))
         0 "-R"
         1 "2,3"
         2 "-V"
         3 "-N"
         4 "-A")))

(deftest make-filter-remove-useless-attributes
  (let [ds (clj-ml.data/make-dataset :foo [:a] [[1] [2]])
        filter (make-filter :remove-useless-attributes {:dataset-format ds :max-variance 95})]
    (is (= (.getMaximumVariancePercentageAllowed filter) 95))))

(deftest make-filter-discretize-sup
  (let [ds (clj-ml.data/make-dataset :test [:a :b {:c [:g :m]}]
                                     [ [1 2 :g]
                                       [2 3 :m]
                                       [4 5 :g]])
        _ (clj-ml.data/dataset-set-class ds 2)
        f (make-filter :supervised-discretize {:dataset-format ds :attributes [0]})]
    (is (= weka.filters.supervised.attribute.Discretize
           (class f)))))

(deftest make-filter-discretize-unsup
  (let [ds (clj-ml.data/make-dataset :test [:a :b {:c [:g :m]}]
                                     [ [1 2 :g]
                                       [2 3 :m]
                                       [4 5 :g]])
        f (make-filter :unsupervised-discretize {:dataset-format ds :attributes [0]})]
    (is (= weka.filters.unsupervised.attribute.Discretize
           (class f)))))

(deftest make-filter-nominal-to-binary-sup
  (let [ds (clj-ml.data/make-dataset :test [:a :b {:c [:g :m]}]
                                     [ [1 2 :g]
                                       [2 3 :m]
                                       [4 5 :g]])
        foo1(clj-ml.data/dataset-set-class ds 2)
        f (make-filter :supervised-nominal-to-binary {:dataset-format ds})]
    (is (= weka.filters.supervised.attribute.NominalToBinary
           (class f)))))

(deftest make-filter-nominal-to-binary-unsup
  (let [ds (clj-ml.data/make-dataset :test [:a :b {:c [:g :m]}]
                                     [ [1 2 :g]
                                       [2 3 :m]
                                       [4 5 :g]])
        f (make-filter :unsupervised-nominal-to-binary {:dataset-format ds :attributes [2]})]
    (is (= weka.filters.unsupervised.attribute.NominalToBinary
           (class f)))))

(deftest make-filter-remove-attributes
  (let [ds (clj-ml.data/make-dataset :test [:a :b {:c [:g :m]}]
                                     [ [1 2 :g]
                                       [2 3 :m]
                                       [4 5 :g]])
        f (make-filter :remove-attributes {:dataset-format ds :attributes [0]})]
    (is (= weka.filters.unsupervised.attribute.Remove
           (class f)))
    (let [res (filter-apply f ds)]
      (is (= (dataset-format res)
             [:b {:c '(:m :g)}])))))

(deftest make-apply-filter-remove-attributes
  (let [ds (clj-ml.data/make-dataset :test [:a :b {:c [:g :m]}]
                                     [ [1 2 :g]
                                       [2 3 :m]
                                       [4 5 :g]])
        res (make-apply-filter :remove-attributes {:attributes [0]} ds)]
    (is (= (dataset-format res)
           [:b {:c '(:m :g)}]))))


(deftest make-apply-filter-numeric-to-nominal
  (let [ds (clj-ml.data/make-dataset :test [:a :b {:c [:g :m]}]
                                     [ [1 2 :g]
                                       [2 3 :m]
                                       [4 5 :g]])
        res (make-apply-filter :numeric-to-nominal {} ds)]
    (is (= (dataset-format res)
           [{:a '(:4 :2 :1)} {:b '(:5 :3 :2)} {:c '(:m :g)}]))))
