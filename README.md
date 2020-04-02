# bd4h-group-project

## Setup
The following setup steps need to be performed every time you launch a new cluster.

### Change the S3 Path
Modify config.py to point to the mimic dataset in your S3 bucket (unless we figure out how to share buckets). 

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

### Install Dependencies on the Workers (Run setup_cluster.sh)
Run the setup_cluster.sh script, which uses parallel ssh to install the python dependencies on the 
workers. 
```
sh setup_cluster.sh
```

## Running the Application
```
spark-submit main.py
```
TODO: add instructions for testing specific functions once we have unit tests ready
