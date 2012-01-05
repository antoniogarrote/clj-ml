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

(deftest make-instance-nils
  (let [dataset (make-dataset :test
                              [:a :b]
                              1)
        inst (make-instance dataset [1 nil])]
  (is (= (class inst)
         weka.core.Instance))
  (is (= 2 (.numValues inst)))
  (is (= 1.0 (.value inst 0)))
  (is (Double/isNaN (.value inst 1)))))

(deftest dataset-make-dataset-with-default-class
  (let [ds (clj-ml.data/make-dataset :test [:a :b {:c [:d :e]}] [] {:class :c})
        ds2 (clj-ml.data/make-dataset :test [:a :b {:c [:d :e]}] [] {:class 2})]
    (is (= (clj-ml.data/dataset-class-name ds)
           :c))
    (is (= (clj-ml.data/dataset-class-index ds2)
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
      (is (= 0 (.classIndex (dataset-set-class dataset "a")))))))

(deftest dataset-class-values-test
  (let [dataset (make-dataset :test
                              [:age :iq {:favorite-color [:red :blue :green]}]
                              [[12 100 :red]
                               [14 110 :blue]
                               [ 25 120 :green]])]
    (testing "when the class is numeric"
      (dataset-set-class dataset :iq)
      (is (= [100.0 110.0 120.0] (dataset-class-values dataset))))
    (testing "when the class is nominal"
      (dataset-set-class dataset :favorite-color)
      (is (= ["red" "blue" "green"] (dataset-class-values dataset))))))

(deftest dataset-name-utils
  (let [dataset (make-dataset :test
                              [:age :iq {:favorite-color [:red :blue :green]}]
                              [[12 100 :red]
                               [14 110 :blue]
                               [ 25 120 :green]])]
    (is (= "test" (dataset-name dataset)))
    (is (= "new-name" (dataset-name (dataset-set-name dataset "new-name"))))
    (is (= "new-name-extra" (dataset-name (dataset-append-name dataset "-extra"))))))

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
  (let [ds (make-dataset "test" [:a :b {:c [:d :e]}] [{:a 1 :b 2 :c nil} [4 nil :e]])]
    (is (= 2 (dataset-count ds)))
    (is (= [{:a 1.0 :b 2.0 :c nil} {:a 4.0 :b nil :c "e"}] (dataset-as-maps ds)))
    (is (= [[1.0 2.0 nil] [4.0 nil "e"]] (dataset-as-vecs ds)))
    (is (= [{:a 1.0 :b 2.0 :c nil} {:a 4.0 :b nil :c "e"}] (map #(instance-to-map %1) (dataset-seq ds))))))

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
    (is (= (first attrs) (attribute-at ds 0) (attribute-at ds :a)))
    (is (= '("a" "b" "c") (map #(.name %) attrs)))
    (is (= '("a" "b" "c") (map #(.name %) (attributes (dataset-at ds 0)))))
    (is (= [(.attribute ds 2)]  (nominal-attributes ds)))
    (is (= [(.attribute ds 0) (.attribute ds 1)]  (numeric-attributes ds)))
    (is (= '(:a :b :c) (attribute-names ds)))))

(deftest replacing-attributes
  (let [ds (make-dataset "test" [:a {:b [:foo :bar]}] [[1 :foo] [2 :bar]])
        _ (dataset-replace-attribute! ds :b (nominal-attribute :b [:baz :shaz]))]
    (is (= [:a {:b [:baz :shaz]}] (dataset-format ds)))))

(deftest dataset-label-helpers
  (let [ds (make-dataset "test" [:a :b {:c [:d :e]}]
                         [{:a 1 :b 2 :c :d} [4 5 :e]])]
    (dataset-set-class ds :c)
    (is (= {:d 0 :e 1} (dataset-class-labels ds) (dataset-labels-at ds :c)))
    (is (= #{:d :e} (attribute-labels (first (nominal-attributes ds)))))))

(deftest dataset-format-and-headers-test
  (let [ds (make-dataset "test" [:a {:b [:foo :bar]}] [[1 :foo] [2 :bar]])]
    (is (= [:a {:b [:foo :bar]}] (dataset-format ds)))
    (let [headers  (headers-only ds)]
      (is (= 0 (dataset-count headers)))
      (is (= "test" (dataset-name headers)))
      (is (= [:a {:b [:foo :bar]}] (dataset-format headers))))))

(deftest dataset-class-helpers
  (let [ds (make-dataset "test" [:a {:b [:foo :bar]}] [[1 :foo] [2 :bar]])]
    (is (= nil (dataset-class-name ds)))
    (dataset-set-class ds :b)
    (is (= :b (dataset-class-name ds)))))

(deftest split-dataset-percentage-test
  (let [ds (make-dataset "test" [:a {:b [:foo :bar]}]
                         [[1 :foo]
                          [2 :bar]
                          [3 :bar]
                          [4 :foo]])
        [a b] (split-dataset ds :percentage 25)]
    (is (= (dataset-count @a) 1))
    (is (= (dataset-count @b) 3))))

(deftest split-dataset-num-test
  (let [ds (make-dataset "test" [:a {:b [:foo :bar]}]
                         [[1 :foo]
                          [2 :bar]
                          [3 :bar]
                          [4 :foo]])
        [a b] (split-dataset ds :num 1)]
    (is (= (dataset-count @a) 1))
    (is (= (dataset-count @b) 3))))

(deftest do-split-dataset-test
  (let [ds (make-dataset "test" [:a {:b [:foo :bar]}]
                         [[1 :foo]
                          [2 :bar]
                          [3 :bar]
                          [4 :foo]])
        [a b] (do-split-dataset ds :percentage 25)]
    (is (= (dataset-count a) 1))
    (is (= (dataset-count b) 3))))

(deftest take-dataset-test
  (let [ds (make-dataset "test" [:a {:b [:foo :bar]}]
                         [[1 :foo]
                          [2 :bar]
                          [3 :bar]
                          [4 :foo]])]
    (is (= (dataset-as-vecs (take-dataset ds 2)) [[1.0 "foo"] [2.0 "bar"]]))))
