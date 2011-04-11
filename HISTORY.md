# clj-ml History

## v0.1.0 - 2011-04-11

### New Features
  * New filter wrappers added: `RemoveUseless` as `:remove-useless`, `Add` as `:add-attribute`
  * New classifier wrappers added: `PaceRegression`, `RandomForest`, M5P Trees and boosted stumps (`LogitBoost`), `AdditiveRegression`, Gradient Boosted Decision Trees, `RotationForest`, `SPegasos`
  * Increased the number of options that can be specified for classifiers.
  * Adds `:clj-streamable` and `:clj-batch` filters which allow for custom
  functions to be provided for filtering the dataset.
  * More idiomatic way and using/applying filters that allows threading.
  * Attribute names can now be specified in the filters and other accessor functions instead of requiring the columns index.
  * New utility functions: `into-fast-vec, dataset-replace-attribute, dataset-class-values, dataset-nominal?, make-apply-filters, classifer-copy-and-train, keyword-name, headers-only, dataset-class-name, attribute-labels-as-strings, dataset-name`
  * Speed improvement in `dataset-as-maps`

### Bug Fixes
  * `is-dataset?` reports falses correctly now.
  * A large ammount of type hinting was added to many areas where it was slowing down real use-cases.
  * Loading and saving instances functions now work with streams (not just files or file names).
  * `nil` values are allowed in datasets (represented as `NaN`s in weka)


## v0.0.4 - 2010-10-28

### New Features
  * Upgraded to Clojure 1.2. (Ben Mabey)
  * Upgraded deps and swithced to modular Incanter dependencies. (Ben Mabey)



## v0.0.3 - 2010-02-28

 * Initial release by Antonio Garrote.
