from pyspark.sql import SparkSession
from pyspark.ml import Pipeline

import sparknlp
from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import Tokenizer
from sparknlp.pretrained import LemmatizerModel

spark = SparkSession.builder.appName('NLP Pipeline').getOrCreate()

df = spark.createDataFrame([
    ("Hello. Look at all this text we're going to process",),
    ("These are exciting times we're living in.",),
    ("Watch what this pipeline can do!",),
    ],
    ['text'])

# DocumentAssembler is a required first step to format df for sparknlp pipeline
doc_assembler = DocumentAssembler().setInputCol('text').setOutputCol('document')

tokenizer = Tokenizer().setInputCols(['document']).setOutputCol('tokens')

lemmatizer_name = 'lemma_antbnc' # John Snow Labs' pretrained English lemmatizer
lang = 'en'
lemmatizer = (LemmatizerModel.pretrained(lemmatizer_name, lang=lang)
        .setInputCols(['tokens'])
        .setOutputCol('lemmas')
        )

finisher = Finisher().setInputCols(['lemmas'])

pipe = Pipeline(stages=[
    doc_assembler,
    tokenizer,
    lemmatizer,
    finisher,
    ])

pipe.fit(df).transform(df).show(truncate=False)
