(ns clj-ml.kernel-functions-test
  (:use [clj-ml kernel-functions] :reload-all)
  (:use [clojure.test]))

(deftest make-kernel-function-polynomic
  (let [kernel (clj-ml.kernel-functions/make-kernel-function :polynomic {:exponent 0.3})
        options (.getOptions kernel)]
    (is (= (aget options 2)
           "-E"))
    (is (= (aget options 3)
           "0.3"))))

(deftest make-kernel-function-radial-basis
  (let [kernel (clj-ml.kernel-functions/make-kernel-function :radial-basis {:gamma 0.3})
        options (.getOptions kernel)]
    (is (= (aget options 2)
           "-G"))
    (is (= (aget options 3)
           "0.3"))))

(deftest make-kernel-function-string
  (let [kernel (clj-ml.kernel-functions/make-kernel-function :string {:lambda 0})
        options (.getOptions kernel)]
    (is (= (aget options 6)
           "-L"))
    (is (= (aget options 7)
           "0.0"))))
