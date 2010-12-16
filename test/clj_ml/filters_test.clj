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
  (let [ds (make-dataset :foo [:a] [[1] [2]])
        filter (make-filter :remove-useless-attributes {:dataset-format ds :max-variance 95})]
    (is (= (.getMaximumVariancePercentageAllowed filter) 95))))

(deftest make-filter-discretize-sup
  (let [ds (make-dataset :test [:a :b {:c [:g :m]}]
                                     [ [1 2 :g]
                                       [2 3 :m]
                                       [4 5 :g]])
        _ (dataset-set-class ds 2)
        f (make-filter :supervised-discretize {:dataset-format ds :attributes [0]})]
    (is (= weka.filters.supervised.attribute.Discretize
           (class f)))))

(deftest make-filter-discretize-unsup
  (let [ds (make-dataset :test [:a :b {:c [:g :m]}]
                                     [ [1 2 :g]
                                       [2 3 :m]
                                       [4 5 :g]])
        f (make-filter :unsupervised-discretize {:dataset-format ds :attributes [0]})]
    (is (= weka.filters.unsupervised.attribute.Discretize
           (class f)))))

(deftest make-filter-nominal-to-binary-sup
  (let [ds (make-dataset :test [:a :b {:c [:g :m]}]
                                     [ [1 2 :g]
                                       [2 3 :m]
                                       [4 5 :g]])
        foo1(dataset-set-class ds 2)
        f (make-filter :supervised-nominal-to-binary {:dataset-format ds})]
    (is (= weka.filters.supervised.attribute.NominalToBinary
           (class f)))))

(deftest make-filter-nominal-to-binary-unsup
  (let [ds (make-dataset :test [:a :b {:c [:g :m]}]
                                     [ [1 2 :g]
                                       [2 3 :m]
                                       [4 5 :g]])
        f (make-filter :unsupervised-nominal-to-binary {:dataset-format ds :attributes [2]})]
    (is (= weka.filters.unsupervised.attribute.NominalToBinary
           (class f)))))

(deftest make-filter-remove-attributes
  (let [ds (make-dataset :test [:a :b {:c [:g :m]}]
                                     [ [1 2 :g]
                                       [2 3 :m]
                                       [4 5 :g]])
        f (make-filter :remove-attributes {:dataset-format ds :attributes [0]})]
    (is (= weka.filters.unsupervised.attribute.Remove
           (class f)))
    (let [res (filter-apply f ds)]
      (is (= (dataset-format res)
             [:b {:c '(:g :m)}])))))

(deftest make-apply-filter-remove-attributes
  (let [ds (make-dataset :test [:a :b {:c [:g :m]}]
                                     [ [1 2 :g]
                                       [2 3 :m]
                                       [4 5 :g]])
        res (make-apply-filter :remove-attributes {:attributes [0]} ds)]
    (is (= (dataset-format res)
           [:b {:c '(:g :m)}]))))


(deftest make-apply-filter-numeric-to-nominal
  (let [ds (make-dataset :test [:a :b {:c [:g :m]}]
                                     [ [1 2 :g]
                                       [2 3 :m]
                                       [4 5 :g]])]
    (testing "when no attributes are specified"
      (is (= (dataset-format  (make-apply-filter :numeric-to-nominal {} ds))
             [{:a '(:1 :2 :4)} {:b '(:2 :3 :5)} {:c '(:g :m)}])))
    (testing "when attributes are specified by index"
       (is (= (dataset-format  (make-apply-filter :numeric-to-nominal {:attributes [0]} ds))
              [{:a '(:1 :2 :4)} :b {:c '(:g :m)}])))
    (testing "when attributes are specified by name"
       (is (= (dataset-format  (make-apply-filter :numeric-to-nominal {:attributes [:b]} ds))
              [:a {:b '(:2 :3 :5)} {:c '(:g :m)}])))))



(deftest make-apply-filter-add-attribute
  (let [ds (make-dataset :test [:a :b {:c [:g :m]}]
                                     [ [1 2 :g]
                                       [2 3 :m]
                                       [4 5 :g]])
        res (add-attribute ds {:type :nominal, :column 1, :name "pet", :labels ["dog" "cat"]})]
    (is (= (dataset-format res)
           [:a {:pet '(:dog :cat)} :b {:c '(:g :m)}]))))

(deftest make-apply-filters-test
  (let [ds (make-dataset :test [:a :b {:c [:g :m]}]
                                     [ [1 2 :g]
                                       [2 3 :m]
                                       [4 5 :g]])
        res (make-apply-filters
             [[:add-attribute {:type :nominal, :column 1, :name "pet", :labels ["dog" "cat"]}]
              [:remove-attributes {:attributes [:a :c]}]] ds)]
    (is (= (dataset-format res)
           [{:pet '(:dog :cat)} :b]))))

(deftest using-regular-filter-fns-with-threading
  (let [ds (make-dataset :test [:a :b {:c [:g :m]}]
                                     [ [1 2 :g]
                                       [2 3 :m]
                                       [4 5 :g]])
        res (-> ds
                (add-attribute {:type :nominal, :column 1, :name "pet", :labels ["dog" "cat"]})
                (remove-attributes {:attributes [:a :c]}))]
    (is (= (dataset-format res)
           [{:pet '(:dog :cat)} :b]))))

(deftest make-apply-filter-clj-streamable
  (let [ds (make-dataset :test [:a :b {:c [:g :m]}]
                                     [ [1 2 :g]
                                       [2 3 :m]
                                       [4 5 :g]])

        rename-attributes (fn [^weka.core.Instances input-format]
                            (doto (weka.core.Instances. input-format 0)
                              (.renameAttribute 0 "foo")
                              (.renameAttribute 1 "bar")))
        inc-nums (fn [^weka.core.Instance instance]
                   (doto (.copy instance)
                     (.setValue 0 (inc (.value instance 0)))
                     (.setValue 1 (+ (.value instance 0) (.value instance 1)))))
        res (make-apply-filter :clj-streamable
                               {:process inc-nums
                                :determine-dataset-format rename-attributes} ds)]
    (is (= (map instance-to-map (dataset-seq res))
           [{:foo 2 :bar 3 :c "g"}
            {:foo 3 :bar 5 :c "m"}
            {:foo 5 :bar 9 :c "g"}]))))


(deftest make-apply-filter-clj-batch
  (let [ds (make-dataset :test [:a]
                                     [ [1]
                                       [2]
                                       [4]])
        max-diff-attr (weka.core.Attribute. "max-diff")
        add-max-diff-attr (fn [^weka.core.Instances input-format]
                            (doto (weka.core.Instances. input-format 0)
                              (.insertAttributeAt max-diff-attr 1)))
        add-max-diff-values (fn [^weka.core.Instances instances]
                              (let [ds-seq (dataset-seq instances)
                                    a-max (apply max (map #(.value % 0) ds-seq))
                                    result (add-max-diff-attr instances)
                                    add-instance #(.add result %)]
                                (doseq [instance ds-seq]
                                  (-> instance
                                      instance-to-vector
                                      (conj (- a-max (.value instance 0)))
                                      (#(weka.core.Instance. 1 (into-array Double/TYPE %)))
                                      add-instance))
                                result))
        res (clj-batch ds
                       {:process add-max-diff-values
                        :determine-dataset-format add-max-diff-attr})]
    (is (= [{:a 1 :max-diff 3}
            {:a 2 :max-diff 2}
            {:a 4 :max-diff 0}]
             (dataset-as-maps res)))))
