# from https://github.com/PacktPublishing/TensorFlow-Machine-Learning-Cookbook/blob/master/Chapter%2007/text_helpers.py

# Text Helper Functions
# ---------------------------------------
#
# We pull out text helper functions to reduce redundant code


from collections import Counter
import numpy as np


def build_vocab(docs, vocabulary_size, stops):
    counts = Counter()
    for doc in docs:
        counts.update(Counter(doc.split()))
    word2id = {word[0]: i for i, word
               in enumerate(counts.most_common(vocabulary_size-1))
               if not word[0] in stops}
    return(word2id)


def doc2ids(doc, word2id):
    return [word2id[word] for word in doc.split() if word in word2id]


# Generate data randomly (N words behind, target, N words ahead)
def generate_batch_data(docs, batch_size, window_size):
    # Fill up data batch
    batch_data = []
    label_data = []
    while len(batch_data) < batch_size:
        # select random doc to start
        rand_doc_ix = int(np.random.choice(len(docs), size=1))
        rand_doc = docs[rand_doc_ix]

        # Pull out center word of interest for each window
        # and create a tuple for each window
        # For doc2vec we keep LHS window only to predict target word
        batch_and_labels = [(rand_doc[i:i+window_size],
                             rand_doc[i+window_size])
                            for i in range(0, len(rand_doc)-window_size)]
        batch, labels = [list(x) for x in zip(*batch_and_labels)]
        # Add document index to batch!! Remember that we must extract the
        # last index in batch for the doc-index
        batch = [x + [rand_doc_ix] for x in batch]

        # extract batch and labels
        batch_data.extend(batch[:batch_size])
        label_data.extend(labels[:batch_size])
    # Trim batch and label at the end
    batch_data = batch_data[:batch_size]
    label_data = label_data[:batch_size]

    # Convert to numpy array
    batch_data = np.array(batch_data)
    label_data = np.transpose(np.array([label_data]))

    return(batch_data, label_data)
