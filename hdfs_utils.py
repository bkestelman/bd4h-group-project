import hdfs
import config
import os

hdfs_host = 'http://localhost'
webhdfs_port = '14000'
hdfs_user = 'hadoop'
hdfs_client = hdfs.InsecureClient(hdfs_host + ':' + webhdfs_port, user=hdfs_user)


def file_exists(path):
    if config.run_on_aws:
        return hdfs_client.status(path, strict=False) is not None
    else:
        return os.path.exists(path)
