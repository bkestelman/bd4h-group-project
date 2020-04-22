HDFS_CSV_DIR="readmissions"
LOCAL_DATA_DIR="data"
LOCAL_CSV_FILE="readmissions.csv"
HEADER="text,label"
mkdir -p ${LOCAL_DATA_DIR} 
hdfs dfs -get ${HDFS_CSV_DIR} ${LOCAL_DATA_DIR}/${HDFS_CSV_DIR}
echo ${HEADER} | cat - ${LOCAL_DATA_DIR}/${HDFS_CSV_DIR}/* > ${LOCAL_DATA_DIR}/${LOCAL_CSV_FILE} # concatenate header and all the partial csv files
rm -rf ${LOCAL_DATA_DIR}/${HDFS_CSV_DIR} # delete the local copy of the partial csv files
hdfs dfs -rm -r ${HDFS_CSV_DIR}
