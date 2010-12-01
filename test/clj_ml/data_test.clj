(ns clj-ml.data-test
  (:use [clj-ml.data] :reload-all)
  (:use [clojure.test]))

(deftest make-instance-num
  (let [dataset (make-dataset :test
                              [:a :b]
                              1)
        inst (make-instance dataset [1 2])]
  (is (= (class inst)
         weka.core.Instance))
  (is (= 2 (.numValues inst)))
  (is (= 1.0 (.value inst 0)))
  (is (= 2.0 (.value inst 1)))))

(deftest make-instance-ord
  (let [dataset (make-dataset :test
                              [:a {:b [:b1 :b2]}]
                              1)
        inst (make-instance dataset [1 :b1])]
  (is (= (class inst)
         weka.core.Instance))
  (is (= 2 (.numValues inst)))
  (is (= 1.0 (.value inst 0)))
  (is (= "b1" (.stringValue inst 1)))))


(deftest dataset-make-dataset-with-default-class
  (let [ds (clj-ml.data/make-dataset :test [:a :b {:c [:d :e]}] [] {:class :c})
        ds2 (clj-ml.data/make-dataset :test [:a :b {:c [:d :e]}] [] {:class 2})]
    (is (= (clj-ml.data/dataset-get-class ds)
           2))
    (is (= (clj-ml.data/dataset-get-class ds2)
           2))))


(deftest dataset-change-class
  (let [dataset (make-dataset :test
                              [:a :b]
                              2)
         _ (clj-ml.data/dataset-set-class dataset 1)]
    (is (= 1 (.classIndex dataset)))
    (is (= 0 (.classIndex (dataset-set-class dataset 0))))
    (testing "when a string or symbol is passed in"
      (is (= 1 (.classIndex (dataset-set-class dataset "b"))))
      (is (= 0 (.classIndex (dataset-set-class dataset :a)))))))

(deftest dataset-count-1
  (let [dataset (make-dataset :test
                              [:a :b]
                              2)]
    (dataset-add dataset [1 2])
    (is (= 1 (dataset-count dataset)))))

(deftest dataset-add-1
  (let [dataset (make-dataset :test
                              [:a :b]
                              2)]
    (dataset-add dataset [1 2])
    (let [inst (.lastInstance dataset)]
      (is (= 1.0 (.value inst 0)))
      (is (= 2.0 (.value inst 1))))))

(deftest dataset-add-2
  (let [dataset (make-dataset :test
                              [:a :b]
                              2)
        instance (make-instance dataset [1 2])]
    (dataset-add dataset instance)
    (let [inst (.lastInstance dataset)]
      (is (= 1.0 (.value inst 0)))
      (is (= 2.0 (.value inst 1))))))

(deftest dataset-extract-at-1
  (let [dataset (make-dataset :test
                              [:a :b]
                              2)]
    (dataset-add dataset [1 2])
    (let [inst (.lastInstance dataset)]
      (is (= 1.0 (.value inst 0)))
      (is (= 2.0 (.value inst 1)))
      (let [inst-ext (dataset-extract-at dataset 0)]
        (is (= 0 (.numInstances dataset)))
        (is (= 1.0 (.value inst-ext 0)))
        (is (= 2.0 (.value inst-ext 1)))))))

(deftest dataset-pop-1
  (let [dataset (make-dataset :test
                              [:a :b]
                              2)]
    (dataset-add dataset [1 2])
    (let [inst (.lastInstance dataset)]
      (is (= 1.0 (.value inst 0)))
      (is (= 2.0 (.value inst 1)))
      (let [inst-ext (dataset-pop dataset)]
        (is (= 0 (.numInstances dataset)))
        (is (= 1.0 (.value inst-ext 0)))
        (is (= 2.0 (.value inst-ext 1)))))))

(deftest dataset-seq-1
  (let [dataset (make-dataset :test [:a :b {:c [:e :f]}] [[1 2 :e] [3 4 :f]])
        seq (dataset-seq dataset)]
    (is (sequential? seq))))


(deftest working-sequences-and-helpers
  (let [ds (make-dataset "test" [:a :b {:c [:d :e]}] [{:a 1 :b 2 :c :d} [4 5 :e]])]
    (is (= 2 (dataset-count ds)))
    (is (= [{:a 1 :b 2 :c :d} {:a 4 :b 5 :c :e}] (dataset-as-maps ds)))
    (is (= [{:a 1 :b 2 :c :d} {:a 4 :b 5 :c :e}] (map #(instance-to-map %1) (dataset-seq ds))))))

(deftest dataset-instance-predicates
  (let [ds (make-dataset "test" [:a :b {:c [:d :e]}] [{:a 1 :b 2 :c :d} [4 5 :e]])
        inst (dataset-at ds 0)]
    (is (is-dataset? ds))
    (is (not (is-dataset? inst)))
    (is (not (is-dataset? "something else")))
    (is (is-instance? inst))
    (is (not (is-instance? ds)))))


(deftest attributes-tests
  (let [ds (make-dataset "test" [:a :b {:c [:d :e]}] [{:a 1 :b 2 :c :d} [4 5 :e]])
        attrs (attributes ds)]
    (is (every? #(instance? weka.core.Attribute %) attrs))
    (is (= '("a" "b" "c") (map #(.name %) attrs)))
    (is (= '("a" "b" "c") (map #(.name %) (attributes (dataset-at ds 0)))))
    (is (= [(.attribute ds 2)]  (nominal-attributes ds)))
    (is (= [(.attribute ds 0) (.attribute ds 1)]  (numeric-attributes ds)))
    (is (= '(:a :b :c) (attribute-names ds)))))

(deftest replacing-attributes
  (let [ds (make-dataset "test" [:a {:b [:foo :bar]}] [[1 :foo] [2 :bar]])
        _ (dataset-replace-attribute! ds :b (nominal-attribute :b [:baz :shaz]))]
    (is (= [:a {:b [:shaz :baz]}] (dataset-format ds)))))
