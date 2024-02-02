from fastbook import search_images_ddg, download_url
from tqdm import tqdm
from PIL import Image
from time import time
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple,List
import numpy as np
import torch
import logging

logging.basicConfig(level=logging.INFO)

def gather_image_urls(search_queries: List[str]) -> List[str]:
    all_urls = []
    for query in tqdm(search_queries):
        while True:
            try:
                urls = search_images_ddg(query, max_images=200)
                break
            except:
                print(f'refused by duckduckgo, waiting and trying again...')
                time.sleep(2)
        all_urls += urls
    return all_urls


def download_single_image(idx_url_output : List[Tuple[str,str,str]]) -> str | None:
    idx,url,output_folder = idx_url_output
    try:
        download_url(url, f'{output_folder}{idx}', show_progress=False, timeout=2)
        return f'{output_folder}{idx}'
    except:
        return None
    
def download_images_parallel(urls: List[str], output_folder_path, max_threads=10) -> List[str]:
    jobs = []
    for idx,url in tqdm(enumerate(urls)):
        jobs.append((idx,url,output_folder_path))
        
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        return [image_path for image_path in list(tqdm(executor.map(download_single_image, jobs))) if image_path is not None]
    
def batch_encode(image_paths : str, model : SentenceTransformer) -> Tuple[np.ndarray,List[str]]:
    images : List[Image.Image] = []
    cleaned_paths = []
    for path in tqdm(image_paths):
        try:
            images.append(Image.open(path))
            cleaned_paths.append(path)
        except:
            pass
    print(len(images), len(image_paths))
    embeddings = model.encode(images, batch_size=1024, show_progress_bar=True)
    for image in images:
        image.close()
    return embeddings, cleaned_paths
    
def index_data(db_address: str, collection_name: str, embeddings: np.ndarray, payload: List[dict], ids: List[uuid.UUID]):

    client = QdrantClient(db_address)

    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=rest.VectorParams(
            size=512,
            distance=rest.Distance.COSINE,
        )
    )

    client.upload_collection(
        collection_name=collection_name,
        vectors=list(embeddings),
        payload=payload,
        ids=ids
    )

if __name__ == '__main__':
    
    search_queries = [
        'dog memes',
        'animal memes',
        # 'cat memes',
        # 'funny animal memes',
        # 'Cute animal memes',
        # 'Classic animal memes',
        # 'Pet fail memes'
    ]
    
    logging.info('loading model...')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = SentenceTransformer("clip-ViT-B-32", device=device)
    
    logging.info('model loaded, gathering image urls...')
    
    image_urls = gather_image_urls(search_queries)
    
    logging.info('downloading images...')
    
    image_paths = download_images_parallel(image_urls, './meme-images/', max_threads=10)
    
    logging.info('embedding images...')
    
    embeddings, image_paths = batch_encode(image_paths, model)
    
    logging.info('indexing...')

    payload = [{'image_path':image_path} for image_path in image_paths]
    meme_ids = [uuid.uuid4().hex for _ in range(len(embeddings))]
    
    index_data('localhost', 'meme-images', embeddings, payload, meme_ids)
    
    logging.info('preprocessing done!')
    