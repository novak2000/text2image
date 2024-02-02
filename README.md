
# pets meme search


## dataset exploration
Data is downloaded using duckduckGO api for image search.

Data exploration can be found inside [`meme2_analysis.ipynb`](./meme2_analysis.ipynb)


## How to run app:

1. run qdrant-db container
```sh
docker compose up qdrant-db -d
```

2. download, preprocess and insert data into vector database
```sh
pip install -r requirements.txt
python3 preprocess.py

```

3. run streamlit frontend
```sh
cd frontend
streamlit run streamlit.py
```

4. search memes on [localhost:8501](http://localhost:8501)


## query examples

### good query examples

TODO

### bad query examples

TODO

## system evaluation

TODO