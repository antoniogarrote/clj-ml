(defproject clj-ml "0.0.3-SNAPSHOT"
  :description "Machine Learning library for Clojure built around Weka and friends"
  :repositories {"leadtune-repo" "http://c0026236.cdn1.cloudfiles.rackspacecloud.com/repo"}
  :java-source-path "src/java"
  :javac-fork "true"
  :dependencies [[org.clojure/clojure "1.2.0"]
                 [org.clojure/clojure-contrib "1.2.0"]
                 [org.mongodb/mongo-java-driver "1.0"]
                 [incanter/incanter-core "1.2.3"]
                 [incanter/incanter-charts "1.2.3"]
                 [lt/weka "3.6.3"]]
  :dev-dependencies [[autodoc "0.7.0"]
                     [lein-javac "1.2.1-SNAPSHOT"]]
  :autodoc { :name "clj-ml", :page-title "clj-ml machine learning Clojure's style"
             :author "Antonio Garrote <antoniogarrote@gmail.com>"
             :copyright "2010 (c) Antonio Garrote, under the MIT license"})
