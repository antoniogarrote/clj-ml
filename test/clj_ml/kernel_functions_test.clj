(ns clj-ml.kernel-functions-test
  (:use [clj-ml kernel-functions] :reload-all)
  (:use clojure.test midje.sweet))

(deftest make-kernel-function-polynomic
  (fact
    (let [kernel (clj-ml.kernel-functions/make-kernel-function :polynomic {:exponent 0.3})
          options (vec (.getOptions kernel))]
      options => (contains ["-E" "0.3"]))))

(deftest make-kernel-function-radial-basis
  (fact
    (let [kernel (clj-ml.kernel-functions/make-kernel-function :radial-basis {:gamma 0.3})
          options (vec (.getOptions kernel))]
      options => (contains ["-G" "0.3"]))))

(deftest make-kernel-function-string
  (fact
    (let [kernel (clj-ml.kernel-functions/make-kernel-function :string {:lambda 0})
          options (vec (.getOptions kernel))]
      options => (contains ["-L" "0.0"]))))
