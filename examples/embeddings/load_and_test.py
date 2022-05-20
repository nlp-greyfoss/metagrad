from examples.embeddings.utils import load_pretrained, search

if __name__ == '__main__':
    vocab, embeddings = load_pretrained('./word2vec/cbow.vec')

    search('孙悟空', embeddings, vocab)
