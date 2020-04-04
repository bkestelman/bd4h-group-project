### Run the application with spark-submit, adding necessary configuration

PYTHON=python3
PACKAGES='com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.5'

export PYSPARK_PYTHON=${PYTHON}
export PYSPARK_DRIVER_PYTHON=${PYTHON}

spark-submit \
	--packages ${PACKAGES} \
	$@ \
	nlp_pipeline.py
