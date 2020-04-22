from pyspark.sql.functions import col, rand
import conf.config as config

def get_sample(dataset, balance=config.balance_dataset_negatives, balance_ratio=config.balance_dataset_ratio, sample_size=config.sample_size):
    """
    @param dataset : Spark dataframe to take sample from 
    @param balance : If False, take an arbitrary sample from dataset of given sample_size.
        If True, balance the labels according to the given balance_ratio before sampling
    @param balance_ratio
    @param sample_size
    Note: for balancing, we currently assume there are more negative labels than positive labels, so a balance_ratio of e.g. 4  will produce a sample with 4 times as many negatives as positives
    """
    SEED = config.SEED
    if not balance:
        return dataset.limit(sample_size)
    pos_labels = dataset.where(col('LABEL') == 1)
    neg_labels = dataset.where(col('LABEL') == 0)
    neg_percent_subsample = int(pos_labels.count() * balance_ratio) / neg_labels.count()
    neg_labels = neg_labels.sample(withReplacement=False, fraction=neg_percent_subsample, seed=SEED)
    dataset = pos_labels.union(neg_labels).orderBy(rand(seed=SEED))
    return dataset
