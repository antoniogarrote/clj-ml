(defproject clj-ml "0.0.3-SNAPSHOT"
  :description "Machine Learning library for Clojure built around Weka and friends"
  :java-source-path "src/java"
  :javac-fork "true"
  :dependencies [[org.clojure/clojure "1.1.0"]
                 [org.clojure/clojure-contrib "1.1.0"]
                 [lein-javac "0.0.2-SNAPSHOT"]
                 [incanter/incanter-full "1.0-master-SNAPSHOT"]
                 [com.mongodb/mongo "1.0"]
                 [weka/weka "3.6.2"]]
  :dev-dependencies [[autodoc "0.7.0"]]
  :autodoc { :name "clj-ml", :page-title "clj-ml machine learning Clojure's style"
             :author "Antonio Garrote <antoniogarrote@gmail.com>"
             :copyright "2010 (c) Antonio Garrote, under the MIT license"})
