# bd4h-group-project

## Setup
```
sudo yum install -y git
git clone https://github.gatech.edu/mpatel364/bd4h-group-project
cd bd4h-group-project
cat > key.pem # copy-paste your private EC2 key here (needed to allow the master to connect to the workers)
```
Modify cluster_hosts.conf with the hostnames of your workers
```
sh setup_cluster.sh
```

## Running
`spark-submit main.py`
