(defproject com.leadtune/clj-ml "0.1.5-SNAPSHOT"
  :description "Machine Learning library for Clojure built around Weka and friends"
  :repositories {"leadtune-repo" "http://c0026236.cdn1.cloudfiles.rackspacecloud.com/repo"}
  :java-source-path "src/java"
  :javac-fork "true"
  :warn-on-reflection true
  :dependencies [[org.clojure/clojure "1.2.0"]
                 [org.clojure/clojure-contrib "1.2.0"]
                 [org.mongodb/mongo-java-driver "1.0"]
                 [incanter/incanter-core "1.2.3"]
                 [incanter/incanter-charts "1.2.3"]
                 [lt/weka "3.6.3"]
                 [hr.irb/fastRandomForest "0.98"]]
  :dev-dependencies [[midje "1.3-alpha1"]
                     [autodoc "0.7.0"
                      :exclusions [org.clojure/clojure org.clojure/clojure-contrib
                                   ant/ant ant/ant-launcher]] ;; older ant breaks newer lein
                     [lein-javac "1.2.1-SNAPSHOT"
                      :exclusions [org.clojure/clojure]]
                     [swank-clojure "1.2.1"
                      :exclusions [org.clojure/clojure]]]
  ;:hooks [leiningen.hooks.difftest]
  :autodoc { :name "clj-ml", :page-title "clj-ml machine learning Clojure's style"
             :author "Antonio Garrote <antoniogarrote@gmail.com>"
             :copyright "2010 (c) Antonio Garrote, under the MIT license"})
