<div style="position: relative; text-align: center; background-image: url('https://th.bing.com/th/id/OIP.FhY2jL9E3OtyWAmmT_fFaAHaDt?w=341&h=175&c=7&r=0&o=5&dpr=1.5&pid=1.7'); background-size: cover; background-position: center; border-radius: 20px; border: 2px solid #64B5F6; padding: 15px; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.4), 0px 6px 20px rgba(0, 0, 0, 0.19); transform: perspective(1000px) rotateX(5deg) rotateY(-5deg); transition: transform 0.5s ease-in-out;">
    <div style="position: relative; z-index: 1; background-color: rgba(255, 255, 255, 0.9); backdrop-filter: blur(10px); border-radius: 20px; padding: 20px;">
        <h1 style="color: red; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.4); font-weight: bold; margin-bottom: 10px; font-size: 32px;">Welcome!</h1>
        <p style="color: #1976D2; font-size: 18px; margin: 10px 0;">
            I'm Mustafa Shoukat, a Generative Expert. I'm in the world of LLMs and exploring various concepts and techniques to enhance my skills.
        </p>
        <p style="color: #000000; font-size: 16px; font-style: italic; margin: 10px 0;">
            "I am just a humble data practitioner. I make mistakes and I have blind spots. If you notice things I can improve or if you just want to chat, please feel free to DM me or connect :)"
        </p>
        <p style="color: #2980B9; font-size: 16px; font-style: italic; margin: 10px 0;">
            <strong>About Notebook:</strong> 🧠 In this notebook, I'll unlock the potential of Facebook AI Similarity Search 🔍🤖.
        </p>
        <p style="color: #27AE60; font-size: 16px; font-style: italic; margin: 10px 0;">
            This notebook demonstrates the process of using Facebook AI Similarity Search (Faiss) along with Sentence Transformers to create a vector-based similarity search for an NLP dataset. By following the steps outlined, you will learn how to encode text data into embeddings, build a Faiss index, and perform similarity searches to find related text entries.
        </p>
        <h2 style="color: red; margin-top: 15px; font-size: 28px;">Contact Information</h2>
        <table style="width: 100%; margin-top: 15px; border-collapse: collapse;">
            <tr style="background-color: #64B5F6; color: #ffffff;">
                <th style="padding: 8px; border-bottom: 2px solid #000000;">Name</th>
                <th style="padding: 8px; border-bottom: 2px solid #000000;">Email</th>
                <th style="padding: 8px; border-bottom: 2px solid #000000;">LinkedIn</th>
                <th style="padding: 8px; border-bottom: 2px solid #000000;">GitHub</th>
                <th style="padding: 8px; border-bottom: 2px solid #000000;">Kaggle</th>
            </tr>
            <tr style="background-color: #FFFFFF; color: #000000;">
                <td style="padding: 8px;">Mustafa Shoukat</td>
                <td style="padding: 8px;">mustafashoukat.ai@gmail.com</td>
                <td style="padding: 8px;">
                    <a href="https://www.linkedin.com/in/mustafashoukat/" target="_blank">
                        <img src="https://img.shields.io/badge/LinkedIn-0e76a8.svg?style=for-the-badge&logo=LinkedIn&logoColor=white" alt="LinkedIn Badge" style="border-radius: 5px; width: 100px;">
                    </a>
                </td>
                <td style="padding: 8px;">
                    <a href="https://github.com/Mustafa-Shoukat1" target="_blank">
                        <img src="https://img.shields.io/badge/GitHub-171515.svg?style=for-the-badge&logo=GitHub&logoColor=white" alt="GitHub Badge" style="border-radius: 5px; width: 100px;">
                    </a>
                </td>
                <td style="padding: 8px;">
                    <a href="https://www.kaggle.com/mustafashoukat" target="_blank">
                        <img src="https://img.shields.io/badge/Kaggle-20beff.svg?style=for-the-badge&logo=Kaggle&logoColor=white" alt="Kaggle Badge" style="border-radius: 5px; width: 100px;">
                    </a>
                </td>
            </tr>
        </table>
    </div>
</div>




# **FAISS: Facebook AI Similarity Search**

**FAISS** is a library developed by Facebook AI Research for efficient similarity search and clustering of dense vectors. It excels in handling large-scale datasets and is widely used in applications like recommendation systems, image retrieval, and natural language processing.

## Key Features of FAISS

1. **High Performance**: Optimized for speed and memory usage, FAISS can efficiently handle large datasets with millions of vectors.
2. **Versatile Indexing**: Offers various indexing methods such as flat (brute-force), IVF (inverted file system), and HNSW (Hierarchical Navigable Small World) graphs.
3. **GPU Support**: Leverages GPUs to accelerate similarity search, enhancing performance for large-scale datasets.
4. **Quantization Techniques**: Advanced quantization techniques reduce the memory footprint of indexes, enabling efficient storage and retrieval.
5. **Extensibility**: Easily integrates with other machine learning and data processing libraries.

## How FAISS Works

1. **Indexing**: Vectors are indexed using one of the available methods. For instance, a flat index stores all vectors and conducts a brute-force search, while IVF groups vectors into clusters and searches within relevant clusters.
2. **Querying**: Upon receiving a query vector, FAISS searches the index to find the nearest neighbors, with efficiency and speed depending on the chosen indexing method.
3. **Retrieval**: Retrieves and returns the closest vectors based on a chosen distance metric, typically Euclidean or cosine similarity.

## Applications of FAISS

- **Recommendation Systems**: Finding similar items or users for recommendations.
- **Image Retrieval**: Searching for images similar to a query image based on feature vectors.
- **Natural Language Processing**: Finding semantically similar sentences or documents.
- **Anomaly Detection**: Identifying outliers by searching for points without close neighbors.

## Example Usage

Here's a simple example of how to use FAISS to build and query a vector index:

```python
import faiss
import numpy as np

# Create some data
d = 64                           # dimension
nb = 100000                      # database size
nq = 10000                       # number of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

# Build the index
index = faiss.IndexFlatL2(d)     # IndexFlatL2 is a simple brute-force index
print(index.is_trained)
index.add(xb)                    # add vectors to the index
print(index.ntotal)

# Perform a search
k = 4                           # number of nearest neighbors
D, I = index.search(xq, k)      # search for the k nearest neighbors of the query vectors
print(I[:5])                    # indices of the nearest neighbors
print(D[:5])                    # distances to the nearest neighbors

# Explore Given Notebook in this Repository 📓🔍


# Thank you for exploring FAISS with this example! 🎉✨

We hope this demonstration has provided you with a clear understanding of how to utilize FAISS for efficient similarity search and clustering. Whether you're working on recommendation systems, image retrieval, or natural language processing, FAISS offers powerful tools to handle large-scale datasets with ease. 🚀💡

# Explore Given Notebook in this Repository 📓🔍

## Happy searching! 🔎🔬
