chmod 400 key.pem
yum install -y pssh     # Install parallel ssh client
pscp.pssh -i -h cluster_hosts.conf -x "-i key.pem" requirements.txt ~
pssh -i -h cluster_hosts.conf -x "-i key.pem" \
    "sudo mkdir ~/nltk_data; \
    sudo chmod 777 ~/nltk_data; \
    sudo pip install requirements.txt"


