import random
from pyspark import SparkContext
from pyspark import SparkConf

spconfig = SparkConf().setAppName("EstimatePi").setMaster("local")
sc = SparkContext(conf=spconfig)


def inside(p):
    x, y = random.random(), random.random()
    return x*x + y*y < 1


NUM_SAMPLES = 1000000
count = sc.parallelize(range(0, NUM_SAMPLES)).filter(inside).count()
print('Pi is roughly %f' % (4.0 * count / NUM_SAMPLES))
sc.stop()
