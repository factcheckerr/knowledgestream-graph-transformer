import pickle

with open('triples_chunks2/all_triples.pkl', 'rb') as f:
    try:
        obj = pickle.load(f)
        print("file loaded successfully")
    except EOFError:
        print("File is empty or corrupted.")
