#FOR UBUNTU USERS ONLY
sudo apt update
sudo apt install python3-pip -y
pip3 install virtualenv
source ~/.bashrc
virtualenv venv
source venv/bin/activate
pip install biopython==1.77 torch skorch pandas mysql-connector-python matplotlib