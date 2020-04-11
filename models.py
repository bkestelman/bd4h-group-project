from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit
from pyspark.sql.functions import col, when, count
import matplotlib.pyplot as plt

class Model:

    def __init__(self, algorithm, train, test, features_col='FEATURES', label_col='LABEL', **kwargs):

        self.kwargs = kwargs
        # self.cv = kwargs.get('cv', False)
        # self.lr_class_weights = kwargs.get('set_lr_class_weights', False)

        self.features_col = features_col
        self.label_col = label_col
        self.algorithm = algorithm
        self.train = train
        self.test = test
        self.model = self.get_model()
        self.evaluator = BinaryClassificationEvaluator().setLabelCol(label_col)
        self.estimator = self.model
        self.trained_model = None  # will be populated after calling model.fit()

    def get_model(self):

        if self.algorithm == 'LogisticRegression':
            if self.kwargs.get('set_lr_class_weights', False):  # set weights
                weight_col = 'classWeights'
                neg_labels = self.train.where(col('LABEL') == 0).count()
                train_count = self.train.count()
                balancing_ratio = neg_labels / train_count
                print('balancing ratio: {:.2f}'.format(balancing_ratio))
                self.train = self.train.withColumn(weight_col, when(col('LABEL') == 1, balancing_ratio)
                                                   .otherwise(1 - balancing_ratio))
                model = LogisticRegression(featuresCol='FEATURES', labelCol='LABEL', maxIter=5,
                                           weightCol=weight_col)
            else:

                model = LogisticRegression(featuresCol=self.features_col, labelCol=self.label_col, maxIter=5)

            if self.kwargs.get('cv', False):
                param_grid = ParamGridBuilder() \
                    .addGrid(model.fitIntercept, [False, True]) \
                    .addGrid(model.regParam, [0.01, 0.5, 2.0]) \
                    .build()
                # .addGrid(lr.aggregationDepth, [2, 5, 10])\
                # .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
                # .build()
                cv_model = CrossValidator(estimator=model, estimatorParamMaps=param_grid, evaluator=self.evaluator,
                                          numFolds=3)
                return cv_model
            else:
                return model

        raise NotImplemented('algorithm={} not implemented'.format(self.algorithm))

    def train_model(self):
        print('training {}'.format(self.algorithm))
        self.trained_model = self.model.fit(self.train)

    def evaluate(self):
        predict_train = self.trained_model.transform(self.train)
        predictions = self.trained_model.transform(self.test)
        print('Train Area Under ROC', self.evaluator.evaluate(predict_train))
        print('Test Area Under ROC', self.evaluator.evaluate(predictions))

        # training_summary = self.trained_model.summary
        # roc = training_summary.roc.toPandas()
        # plt.plot(roc['FPR'], roc['TPR'])
        # plt.ylabel('False Positive Rate')
        # plt.xlabel('True Positive Rate')
        # plt.title('ROC Curve')
        # plt.show()
        # print('Training set areaUnderROC: ' + str(training_summary.areaUnderROC))
        #
        # pr = training_summary.pr.toPandas()
        # plt.plot(pr['recall'], pr['precision'])
        # plt.ylabel('Precision')
        # plt.xlabel('Recall')
        # plt.show()

        # Train Area Under ROC 0.9999999557292707
        # Test Area Under ROC 0.615184213495825
