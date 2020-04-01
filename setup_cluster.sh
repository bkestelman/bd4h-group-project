chmod 400 key.pem
sudo yum install -y pssh     # Install parallel ssh client
pscp.pssh -h cluster_hosts.conf -x "-oStrictHostKeyChecking=no -i key.pem" requirements.txt ~
pssh -i -h cluster_hosts.conf -x "-oStrictHostKeyChecking=no -i key.pem" \
    "sudo mkdir /home/nltk_data; \
    sudo chmod 777 /home/nltk_data; \
    sudo pip install -U -r requirements.txt"
sudo pip install -U -r requirements.txt

