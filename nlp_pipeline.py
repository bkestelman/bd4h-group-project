from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import Word2Vec

import sparknlp
from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import Tokenizer
from sparknlp.pretrained import LemmatizerModel, BertEmbeddings, ElmoEmbeddings

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

# BERT is a state of the art pretrained word embedding model
# Couldn't test locally - ran out of memory. May be worthwhile testing on the cluster
#word_embeddings = (BertEmbeddings.pretrained('bert_base_cased', lang=lang)
#    .setInputCols(['lemmas'])
#    .setOutputCols('embeddings')
#    )
#word_embeddings = (ElmoEmbeddings.pretrained('elmo', lang=lang)
#        .setInputCols(['lemmas'])
#        .setOutputCols('embeddings')
#    )

finisher = Finisher().setInputCols(['lemmas'])

word_vector_size=5
word_embeddings = Word2Vec(vectorSize=word_vector_size, minCount=0, inputCol='finished_lemmas', outputCol='embeddings')

pipe = Pipeline(stages=[
    doc_assembler,
    tokenizer,
    lemmatizer,
    finisher,
    word_embeddings,
    ])

model = pipe.fit(df)
#model = word_embeddings.fit(processed)
model.stages[4].getVectors().show(truncate=False)
model.transform(df).show(truncate=False)
