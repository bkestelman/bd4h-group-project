import hdfs

hdfs_host = 'http://localhost'
webhdfs_port = '14000'
hdfs_user = 'hadoop'
hdfs_client = hdfs.InsecureClient(hdfs_host + ':' + webhdfs_port, user=hdfs_user)

def file_exists(path):
    return hdfs_client.status(path, strict=False) is not None
