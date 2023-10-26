from pymilvus import Milvus, DataType, CollectionSchema, FieldSchema
from typing import List

milvus = None  # Declare a global variable

async def init_milvus():
    global milvus
    milvus = Milvus(host='127.0.0.1', port='19530')
    
    # Define the schema for the collection
    dim = 128  # Dimension of the vector, adjust accordingly
    collection_name = "text_collection"  # Name for the collection
    
    id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
    vector_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim, description="Vector field")
    
    fields = [id_field, vector_field]  # List of fields
    
    # schema = CollectionSchema(fields=fields, description="text_collection")  # Pass the list of fields here
    
    # Create the collection
    milvus.create_collection(collection_name, fields=fields)

async def close_milvus():
    global milvus
    milvus.close()

# Assuming milvus is the initialized Milvus client
def store_vectors(collection_name: str, vectors: List[List[float]]):
    # Your logic to store vectors in Milvus
    # Returns IDs of the stored vectors
    status, ids = milvus.insert(collection_name, records=vectors)
    if status.code != 0:
        raise Exception("Failed to insert vectors into Milvus: {}".format(status.message))
    return ids
