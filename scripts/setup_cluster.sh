### Installs pip dependencies on all hosts in the cluster
# Prerequisites: The master needs to be able to ssh to the workers. Make sure you've set up 
# agent-forwarding as described in the README. 

## Create list of worker hosts on the cluser (AWS EMR specific)
CLUSTER_HOSTS_FILE=cluster_hosts.txt
cluster_id=$(aws emr list-clusters | sed '/Id/!d' | awk -F'"' '{print $ 4}' | head -n 1) # get the cluster id
aws emr list-instances --cluster-id ${cluster_id} | sed '/PrivateDnsName/!d' | awk -F'"' '{print $4}' > ${CLUSTER_HOSTS_FILE} # get each host on the cluster's DNS name and output to cluster_hosts.txt
sed -i "/.*$(hostname).*/d" ${CLUSTER_HOSTS_FILE} # remove the master host from the list

## Set up dependencies across all hosts
sudo yum install -y pssh # Install parallel ssh client

BIN=/usr/bin
PIP=${BIN}/$(ls ${BIN} | grep -e 'pip.*3') # Use pip3

pscp.pssh -h ${CLUSTER_HOSTS_FILE} -x "-oStrictHostKeyChecking=no" requirements.txt ~ 	# Copy requirements.txt to worker hosts
# pip install + do any additional setup required by specific packages
pssh -i -h ${CLUSTER_HOSTS_FILE} -x "-oStrictHostKeyChecking=no" \
    "sudo ${PIP} install -U -r requirements.txt"
sudo ${PIP} install -U -r requirements.txt # pip install on master

## Set up requirements for pytorch
sudo ${PIP} install -r requirements_pytorch.txt
sudo python3 -m spacy download en
