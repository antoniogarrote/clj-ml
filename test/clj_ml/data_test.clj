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
        foo(clj-ml.data/dataset-set-class dataset 1)]
    (is (= 1 (.classIndex dataset)))
    (is (= 0 (.classIndex (dataset-set-class dataset 0))))))

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


(deftest working-sequences
  (let [ds (make-dataset "test" [:a :b {:c [:d :e]}] [{:a 1 :b 2 :c :d} [4 5 :e]])]
    (is (= 2 (dataset-count ds)))
    (let [dsm (map #(instance-to-map %1) (dataset-seq ds))]
      (is (= 2 (count dsm)))
      (is (= 1.0 (:a (first dsm))))
      (let [dsb (make-dataset "test" [:a :b {:c [:d :e]}] dsm)]
        (is (= 2 (dataset-count dsb)))))))

(deftest dataset-instance-predicates
  (let [ds (make-dataset "test" [:a :b {:c [:d :e]}] [{:a 1 :b 2 :c :d} [4 5 :e]])
        inst (dataset-at ds 0)]
    (is (is-dataset? ds))
    (is (not (is-dataset? inst)))
    (is (is-instance? inst))
    (is (not (is-instance? ds)))))
