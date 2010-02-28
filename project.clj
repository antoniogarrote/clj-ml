(defproject clj-ml "0.0.3-SNAPSHOT"
  :description "Machine Learning library for Clojure built around Weka and friends"
  :java-source-path "src/java"
  :javac-fork "true"
  :dependencies [[org.clojure/clojure "1.1.0"]
                 [org.clojure/clojure-contrib "1.1.0"]
                 [lein-javac "0.0.2-SNAPSHOT"]
                 [incanter/incanter-full "1.0-master-SNAPSHOT"]
                 [weka/weka "3.6.2"]])
