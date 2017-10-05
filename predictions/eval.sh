cat $1 \
| awk '{print $NF}' \
| paste ../data/emerging.test.conll - \
| python wnuteval.py
