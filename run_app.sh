### Run the application with spark-submit
# Adds necessary configuration such as python version and maven dependencies
# You can still pass options to spark-submit by passing them to this script

# Config
APP=main.py
PYTHON=python3
SPARK_NLP_PKG='com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.5'
PACKAGES=${SPARK_NLP_PKG} # packages should be comma-separated with no whitespace

# Set python version
export PYSPARK_PYTHON=${PYTHON}
export PYSPARK_DRIVER_PYTHON=${PYTHON}

# Run app
spark-submit --packages ${PACKAGES} $@ ${APP}
