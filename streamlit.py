import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from itertools import cycle

st.set_page_config(
    layout="wide",
    page_title = 'test2image',
    initial_sidebar_state='expanded'
)

client = QdrantClient('localhost')
model = SentenceTransformer("clip-ViT-B-32", device='cuda')

def search(query: str):
    search_result = client.search(
        collection_name="ads-images",
        query_vector= model.encode(query)
    )
    return [data.payload['image_path'] for data in search_result]
    
    # return ['./data/65631.jpg',
    #         './data/90900.jpg',
    #         './data/94890.jpg',
    #         './data/90610.jpg',
    #         './data/164311.jpg',
    #         './data/55901.jpg'
    #         ]




st.text_input('image search', key="query")


results = search(st.session_state.query)

column_cnt = 1
cols = cycle(st.columns(column_cnt)) # st.columns here since it is out of beta at the time I'm writing this
for idx, filteredImage in enumerate(results):
    if idx%column_cnt==0:
        st.markdown("""---""")
        cols = cycle(st.columns(column_cnt))
        
    next(cols).image(filteredImage, width = 300, caption=f'{idx+1}')



# for result in results:
#     st.image(result,)
