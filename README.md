# bd4h-group-project

## Setup
The following setup steps need to be performed every time you launch a new cluster.

Note:
- python3 is now required for sparknlp (this is taken care of in the setup_cluster.sh and run_app.sh scripts).
- Run the application using run_app.sh instead of spark-submit for it to be configured correctly. You can still pass options to spark-submit by passing them to run_app.sh instead. 
- The instructions here and the setup scripts are specifically for running on AWS

### Set up SSH Agent-Forwarding
We need to ensure the master host can ssh to the workers (required to install the Spark 
application's python dependencies). 

One way to do this is to copy your EC2 private key to the master, but it's more secure to use ssh 
agent-forwarding instead. To set this up, run the following on your home computer (e.g. from a git 
bash shell):
```
eval $(ssh-agent)	# starts ssh-agent, which is a tool that manages keys and agent-forwarding
ssh-add $EC2_KEY	# replace $EC2_KEY with the path to your EC2 key file
ssh -A -i $EC2_KEY $MASTER_DNS	# replace $MASTER_DNS. The -A option activates agent-forwarding
```
Since we have added the key to ssh-agent and forwarded the agent to our master, the master will be 
able to use this agent to ssh to the workers without having to store the key on the master. 

NOTE: This works out of the box if you use git bash shell (https://git-scm.com/downloads) to ssh 
(tested on Windows). If you're using another ssh client or a plain terminal, you may need to figure
out how to set up ssh-agent or agent-forwarding there. 

### Clone the Git Repo
Once you have ssh'd to the master with agent-forwarding set up, clone the git repo:
```
sudo yum install -y git
git clone https://github.gatech.edu/mpatel364/bd4h-group-project
cd bd4h-group-project
```

### Change the S3 Path
Modify conf/config.py to point to the mimic dataset in your S3 bucket (unless we figure out how to share buckets). 

### Install Dependencies on the Workers (Run setup_cluster.sh)
Run the setup_cluster.sh script, which uses parallel ssh to install the python dependencies on the 
workers. 
```
sh scripts/setup_cluster.sh
```

## Running the Application
```
sh scripts/run_app.sh
```
The above runs the main Spark app, which includes ETL preprocessing to extract labels as well as 
SparkML and sparknlp pipelines. It also writes the labeled data to csv files in HDFS for use by
ML algorithms that run outside of Spark (e.g. our pytorch CNN). 

To download the csv from HDFS to the local filesystem and concatenate the partial files:
```
sh scripts/download_and_cat_hdfs_csv.sh
```
Note this will delete the partial files from HDFS

Now we can run our pytorch model:
```
python3 pytorchmain.py
```

## Using a model trained in a previous run
If saving a model is configured in conf/config.py (see save_model_paths), models will be saved to hdfs and loaded from there automatically.

But if we want to load a model on the first run after launching a cluster, we have to upload it to hdfs. 

For example, here we have the model BasicWord2Vec.Model.5000.tar.gz, which is a Word2VecModel trained on 5000 discharge summaries. To use it, extract it and put it on hdfs:
```
tar -xzvf BasicWord2Vec.Model.5000.tar.gz # should output BasicWord2Vec.Model/
hdfs dfs -put BasicWord2Vec.Model
```
Then when you run the app, it will load this model instead of training from scratch. 

## Run tests
We install pytest as part of the requirements.txt
```
python3 -m pytest
```
