(defproject com.leadtune/clj-ml "0.2.3"
  :description "Machine Learning library for Clojure built around Weka and friends"
  :repositories {"leadtune-repo" "http://utility.leadtune.com:8081/nexus/content/repositories/thirdparty/"}
  :aoc "src/java"
  :javac-fork "true"
  :warn-on-reflection true
  :dependencies [[org.clojure/clojure "1.3.0"]
                 [incanter/incanter-core "1.3.0"]
                 [incanter/incanter-charts "1.3.0"]
                 [lt/weka "3.6.3"]
                 [hr.irb/fastRandomForest "0.98"]]
  :dev-dependencies [[midje "1.3.0-RC2"]
                     [swank-clojure "1.3.4"]])
