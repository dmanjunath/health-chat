### Health Chat
```
# create venv
python3 -m venv .

# enter venv
source bin/activate

# install dependencies
pip install -r requirements.txt

# setup env vars
cp .env.template .env
# fill out the values in .env

# run the script
python3 src/main.py
```