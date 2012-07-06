(ns #^{:author "Ben Mabey <ben@benmabey.com>"}
  clj-ml.attribute-selection
  ""
  (:use [clj-ml data utils options-utils])
  (:import weka.core.OptionHandler
           [weka.attributeSelection
            ASEvaluation ASSearch
            InfoGainAttributeEval GainRatioAttributeEval OneRAttributeEval ReliefFAttributeEval
            SymmetricalUncertAttributeEval ChiSquaredAttributeEval
            AttributeSelection CfsSubsetEval
            ;; search
            GreedyStepwise BestFirst GeneticSearch Ranker RankSearch LinearForwardSelection]))

(defmulti  #^{:skip-wiki true}
  make-obj-options
  "Creates the right parameters for a weka object. Returns a clojure vector."
  (fn [kind map] kind))

;TODO: consider passing in the make-filter-options body here as well in additon to the docstring.
#_(defmacro defsearch
  "Defines the filter's fn that creates a fn to make and apply the filter."
  [filter-name]
  (let [search-keyword (keyword filter-name)]
    `(do
       (defn ~search-name
         ([ds#]
            (make-apply-filter ~filter-keyword {} ds#))
         ([ds# attributes#]
            (make-apply-filter ~filter-keyword attributes# ds#))))))

(defmethod make-obj-options :greedy
;;   -C
;;  Use conservative forward search
;;
;; -B
;;  Use a backward search instead of a
;;  forward one.
;;
;; -P <start set>
;;  Specify a starting set of attributes.
;;  Eg. 1,3,5-7.
;;
;; -R
;;  Produce a ranked list of attributes.
;;
;; -T <threshold>
;;  Specify a theshold by which attributes
;;  may be discarded from the ranking.
;;  Use in conjuction with -R
;;
;; -N <num to select>
;;  Specify number of attributes to select

  ([kind m]
     (let [weka-opts (->> (extract-attributes "-P" :starting-attributes)
                          (check-options m
                                         {:generate-rankings "-R"})
                          (check-option-values m
                                               {:threshold "-T"
                                                :num-attributes "-N"}))]
          (case (m :direction)
             :forward weka-opts
             :conservative-forward (conj weka-opts "-C")
             :backward (conj weka-opts "-B")
             weka-opts))))


;; Sketch of what would be nice to have...
;;(defweka-constructor :greedy GreedyStepwise
;;  -R :generate-rankings
;;  "Produce a ranked list of attributes."
;;  -T <threshold> :threshold
;;  "Specify a theshold by which attributes may be discarded from the ranking. Use in conjuction with :generate-rankings"
;;  -P <start set> :starting-attributes (fn [{:keys [flag alias] :as opts}] (extract-attributes flag alias opts))
;;  "Specify a starting set of attributes."
;;  -C :direction ...)

(defmethod make-obj-options :linear-forward
;; LinearForwardSelection:
;;
;; Extension of BestFirst. Takes a restricted number of k attributes into account. Fixed-set selects a fixed number k of attributes, whereas k is increased in each step when fixed-width is selected. The search uses either the initial ordering to select the top k attributes, or performs a ranking (with the same evalutator the search uses later on). The search direction can be forward, or floating forward selection (with opitional backward search steps).
;;
;; For more information see:
;;
;; Martin Guetlein (2006). Large Scale Attribute Selection Using Wrappers. Freiburg, Germany.
;;
;; Valid options are:
;;
;;  -P <start set>
;;   Specify a starting set of attributes.
;;   Eg. 1,3,5-7.
;;
;;  -D <0 = forward selection | 1 = floating forward selection>
;;   Forward selection method. (default = 0).
;;
;;  -N <num>
;;   Number of non-improving nodes to
;;   consider before terminating search.
;;
;;  -I
;;   Perform initial ranking to select the
;;   top-ranked attributes.
;;
;;  -K <num>
;;   Number of top-ranked attributes that are
;;   taken into account by the search.
;;
;;  -T <0 = fixed-set | 1 = fixed-width>
;;   Type of Linear Forward Selection (default = 0).
;;
;;  -S <num>
;;   Size of lookup cache for evaluated subsets.
;;   Expressed as a multiple of the number of
;;   attributes in the data set. (default = 1)
;;
;;  -Z
;;   verbose on/off
  ([kind m]
     (let [weka-opts (->>
                      (extract-attributes "-P" :starting-attributes)
                      (check-options m {:perform-initial-ranking "-I"})
                      (check-option-values m
                                           {:num-non-inproving "-N"
                                            :num-attrs-in-search "-K"
                                            :subset-eval-cache-size "-S"}))]
         (conj weka-opts "-D" (case (m :direction)
                                  :backward "0"
                                  :forward "1"
                                  :bi-directional "2"
                                  "1"))
       )))

(defmethod make-obj-options :best-first
  ;; BestFirst:
  ;; Searches the space of attribute subsets by greedy hillclimbing augmented with a backtracking facility. Setting the
  ;; number of consecutive non-improving nodes allowed controls the level of backtracking done. Best first may start with
  ;; the empty set of attributes and search forward, or start with the full set of attributes and search backward, or start
  ;; at any point and search in both directions (by considering all possible single attribute additions and deletions at
  ;; a given point).
  ;;
  ;; Valid options are:
  ;;
  ;;  -P <start set>
  ;;   Specify a starting set of attributes.
  ;;   Eg. 1,3,5-7.
  ;;
  ;;  -D <0 = backward | 1 = forward | 2 = bi-directional>
  ;;   Direction of search. (default = 1).
  ;;
  ;;  -N <num>
  ;;   Number of non-improving nodes to
  ;;   consider before terminating search.
  ;;
  ;;  -S <num>
  ;;   Size of lookup cache for evaluated subsets.
  ;;   Expressed as a multiple of the number of
  ;;   attributes in the data set. (default = 1)
   ([kind m]
     (let [weka-opts (->> (extract-attributes "-P" :starting-attributes)
                          (check-option-values m
                                               {:num-non-inproving "-N"
                                                :subset-eval-cache-size "-S"}))]
       (conj weka-opts "-D" (case (m :direction)
                                  :backward "0"
                                  :forward "1"
                                  :bi-directional "2"
                                  "1")))))

(defmethod make-obj-options :genetic
;; GeneticSearch:
;;
;; Performs a search using the simple genetic algorithm described in Goldberg (1989).
;;
;; For more information see:
;;
;; David E. Goldberg (1989). Genetic algorithms in search, optimization and machine learning. Addison-Wesley.
;;
;; BibTeX:
;;
;;  @book{Goldberg1989,
;;     author = {David E. Goldberg},
;;     publisher = {Addison-Wesley},
;;     title = {Genetic algorithms in search, optimization and machine learning},
;;     year = {1989},
;;     ISBN = {0201157675}
;;  }
;;
;;
;; Valid options are:
;;
;;  -P <start set>
;;   Specify a starting set of attributes.
;;   Eg. 1,3,5-7.If supplied, the starting set becomes
;;   one member of the initial random
;;   population.
;;
;;  -Z <population size>
;;   Set the size of the population (even number).
;;   (default = 20).
;;
;;  -G <number of generations>
;;   Set the number of generations.
;;   (default = 20)
;;
;;  -C <probability of crossover>
;;   Set the probability of crossover.
;;   (default = 0.6)
;;
;;  -M <probability of mutation>
;;   Set the probability of mutation.
;;   (default = 0.033)
;;
;;  -R <report frequency>
;;   Set frequency of generation reports.
;;   e.g, setting the value to 5 will
;;   report every 5th generation
;;   (default = number of generations)
;;
;;  -S <seed>
;;   Set the random number seed.
;;   (default = 1)
   ([kind m]
      (->> (extract-attributes "-P" :starting-attributes)
           (check-option-values m
                                {:population-size "-Z"
                                 :num-generations "-G"
                                 :crossover-prob "-C"
                                 :mutation-prob "-M"
                                 :report-freq "-R"
                                 :random-seed "-S"}))))

(defmethod make-obj-options :cfs-subset-eval
  ;; CfsSubsetEval :
  ;;
  ;; Evaluates the worth of a subset of attributes by considering the individual predictive ability of each feature along with the degree of redundancy between them.
  ;;
  ;; Subsets of features that are highly correlated with the class while having low intercorrelation are preferred.
  ;;
  ;; For more information see:
  ;;
  ;; M. A. Hall (1998). Correlation-based Feature Subset Selection for Machine Learning. Hamilton, New Zealand.
  ;;
  ;; BibTeX:
  ;;
  ;;  @phdthesis{Hall1998,
  ;;     address = {Hamilton, New Zealand},
  ;;     author = {M. A. Hall},
  ;;     school = {University of Waikato},
  ;;     title = {Correlation-based Feature Subset Selection for Machine Learning},
  ;;     year = {1998}
  ;;  }
  ;;
  ;;
  ;; Valid options are:
  ;;
  ;;  -M
  ;;   Treat missing values as a separate value.
  ;;
  ;;  -L
  ;;  Don't include locally predictive attributes.
   ([kind m]
      (check-options m
                     {:treat-missing-vals-separate "-M"
                      :ignore-locally-predictive-attrs "-L"})))

(defn attribute-eval-options [m]
;; Valid options are:
;;
;;  -M
;;   treat missing values as a seperate value.
;;
;;  -B
;;   just binarize numeric attributes instead
;;   of properly discretizing them.
   (check-options m
                     {:treat-missing-vals-separate "-M"
                      :binarize-numeric-attrs "-B"}))

(defmethod make-obj-options :info-gain
;;   InfoGainAttributeEval :
;;
;; Evaluates the worth of an attribute by measuring the information gain with respect to the class.
;;
;; InfoGain(Class,Attribute) = H(Class) - H(Class | Attribute).
  ([kind m]
     (attribute-eval-options m)))

(defmethod make-obj-options :chi-squared
;; ChiSquaredAttributeEval :
;;
;; Evaluates the worth of an attribute by computing the value of the chi-squared statistic with respect to the class.
  ([kind m]
     (attribute-eval-options m)))

(defmethod make-obj-options :gain-ratio
  ;; GainRatioAttributeEval :
  ;; Evaluates the worth of an attribute by measuring the gain ratio with respect to the class.
  ;;
  ;; GainR(Class, Attribute) = (H(Class) - H(Class | Attribute)) / H(Attribute).
  ([kind m]
      (check-options m
                     {:treat-missing-vals-separate "-M"})))


(defmethod make-obj-options :symmetrical-uncert
  ;; SymmetricalUncertAttributeEval :
  ;;
  ;; Evaluates the worth of an attribute by measuring the symmetrical uncertainty with respect to the class.
  ;;
  ;; SymmU(Class, Attribute) = 2 * (H(Class) - H(Class | Attribute)) / H(Class) + H(Attribute).
  ;;
  ([kind m]
      (check-options m
                     {:treat-missing-vals-separate "-M"})))

(defmethod make-obj-options :relief
 ;; ReliefFAttributeEval :
 ;;
 ;; Evaluates the worth of an attribute by repeatedly sampling an instance and considering the value of the given attribute for the nearest instance of the same and different class. Can operate on both discrete and continuous class data.
 ;; -M <num instances>
 ;;  Specify the number of instances to
 ;;  sample when estimating attributes.
 ;;  If not specified, then all instances
 ;;  will be used.
 ;;
 ;; -D <seed>
 ;;  Seed for randomly sampling instances.
 ;;  (Default = 1)
 ;;
 ;; -K <number of neighbours>
 ;;  Number of nearest neighbours (k) used
 ;;  to estimate attribute relevances
 ;;  (Default = 10).
 ;;
 ;; -W
 ;;  Weight nearest neighbours by distance
 ;;
 ;; -A <num>
 ;;  Specify sigma value (used in an exp
 ;;  function to control how quickly
 ;;  weights for more distant instances
 ;;  decrease. Use in conjunction with -W.
 ;;  Sensible value=1/5 to 1/10 of the
 ;;  number of nearest neighbours.
 ;;  (Default = 2)
   ([kind m]
      (->> (extract-attributes "-P" :starting-attributes)
           (check-options {:weight "-W"})
           (check-option-values m
                                {:num-instances "-M"
                                 :random-seed "-D"
                                 :number-of-neighbors "-K"
                                 :weight-sigma "-A"}))))

(defmethod make-obj-options :ranker
;; Ranker :
;;
;; Ranks attributes by their individual evaluations. Use in conjunction with attribute evaluators (ReliefF, GainRatio, Entropy etc).
;;
;; Valid options are:
;;
;;  -P <start set>
;;   Specify a starting set of attributes.
;;   Eg. 1,3,5-7.
;;   Any starting attributes specified are
;;   ignored during the ranking.
;;
;;  -T <threshold>
;;   Specify a theshold by which attributes
;;   may be discarded from the ranking.
;;
;;  -N <num to select>
;;   Specify number of attributes to select
  ([kind m]
     (->> (extract-attributes "-P" :starting-attributes)
          (check-option-values m
                               {:threshold "-T"
                                :num-attributes "-N"}))))



(defmethod make-obj-options :one-R
;; OneRAttributeEval :
;;
;; Evaluates the worth of an attribute by using the OneR classifier.
;;
;; Valid options are:
;;
;;  -S <seed>
;;   Random number seed for cross validation
;;   (default = 1)
;;
;;  -F <folds>
;;   Number of folds for cross validation
;;   (default = 10)
;;
;;  -D
;;   Use training data for evaluation rather than cross validaton
;;
;;  -B <minimum bucket size>
;;   Minimum number of objects in a bucket
;;   (passed on to OneR, default = 6)
  ([kind m]
     (->> (check-options m {:use-training-data-for-eval "-D"})
          (check-option-values m
                               {:random-seed "-S"
                                :folds "-N"
                                :bucket-size "-B"}))))

(def obj-aliases
  "Mapping of cjl-ml keywords to actual Weka classes"
  {
   ;; Searches
   :greedy GreedyStepwise
   :best-first BestFirst
   :genetic GeneticSearch
   :ranker Ranker
   :linear-forward LinearForwardSelection
   ;; Evals
   :cfs-subset-eval CfsSubsetEval
   :info-gain InfoGainAttributeEval
   :gain-ratio GainRatioAttributeEval
   :symmetrical-uncert SymmetricalUncertAttributeEval
   :chi-squared ChiSquaredAttributeEval
   :one-R OneRAttributeEval
   :relief ReliefFAttributeEval
   })

(defn make-weka-obj [kind options]
  (let [^OptionHandler f (.newInstance  (kind obj-aliases))]
    (when (not (empty? options))
      (.setOptions f (into-array String (make-obj-options kind options))))
    f))

(defn greedy [& {:as options}]
  (make-weka-obj :greedy options))

(defn best-first [& {:as options}]
  (make-weka-obj :best-first options))

(defn genetic [& {:as options}]
  (make-weka-obj :genetic options))

(defn cfs-subset-eval [& {:as options}]
  (make-weka-obj :cfs-subset-eval options))

(defn symmetrical-uncert [& {:as options}]
  (make-weka-obj :symmetrical-uncert options))

(defn info-gain [& {:as options}]
  (make-weka-obj :info-gain options))

(defn gain-ratio [& {:as options}]
  (make-weka-obj :gain-ratio options))

(defn chi-squared [& {:as options}]
  (make-weka-obj :chi-squared options))

(defn one-R [& {:as options}]
  (make-weka-obj :one-R options))

(defn relief [& {:as options}]
  (make-weka-obj :relief options))

(defn ranker [& {:as options}]
  (make-weka-obj :ranker options))

(defn rank-search
  [& {:keys [evaluator step-size start-point] :as opts}]
  (let [^RankSearch search (RankSearch.)]
    (when evaluator (.setAttributeEvaluator search evaluator))
    (when step-size (.setStepSize search step-size))
    (when start-point (.setStartPoint search start-point))
    search))

(defn attribute-selector
  "Returns an attibute selector.  A search and evaluator object are required.  You may
   also specify that a cross-validation needs to be done by passing in the number of
   folds."
  [& {:keys [search evaluator folds random-seed rank] :as opts}]
  (let [^AttributeSelection attr-sel (AttributeSelection.)]
    (when folds (doto attr-sel (.setXval true) (.setFolds folds)))
    (when random-seed (doto attr-sel (.setSeed random-seed)))
    (when rank (doto attr-sel (.setRanking true)))
    (doto attr-sel
      (.setEvaluator evaluator)
      (.setSearch search))))

(defn apply-opts [f opts]
  (apply f (-> opts vec flatten)))

(defn select-attributes
  "Takes a data set and a selector, or options to build a selector.  The best attributes
   are then selected using the selector's settings.  A list of attributes will be returned
   along with the used selector in the metadata of the list."
  [ds & {:keys [selector search evaluator] :as opts}]
  (let [^AttributeSelection selector (or selector (apply-opts attribute-selector opts))]
    (.SelectAttributes selector ds)
    (with-meta (map #(->> % (dataset-attribute-at ds) keyword-name)
                    (.selectedAttributes selector))
      (merge {:selector selector}
             (zipmap [:search :evaluator] [search (str evaluator)]))))) ;; str evaluator to avoid potential memory leaks

(defn desc [a b]
  "Comparator for descending order"
  (compare b a))

(defn rank-attributes
  "Similar to what select-attributes with ranker would do but it returns the score
   associated with each attribute."
  [ds ^ASEvaluation evaluator]
  (.buildEvaluator evaluator ds)
  (sort-by second desc
           (map #(list (keyword-name %)
                       (.evaluateAttribute evaluator (.index %)))
                (attributes ds))))

