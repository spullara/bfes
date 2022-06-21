# bfes

## Brute force embedding search

Given a set of embeddings, and a query embedding, this algorithm searches for the
nearest embeddings for each query. As it is a brute force search, you don't
need to regenerate the index and can add new embeddings to the index at any
time. The algorithm is O(n) in time and space.