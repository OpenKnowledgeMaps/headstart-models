import tensorflow as tf
import numpy as np
import os
import pickle
from corpora import helpers
from nltk.corpus import stopwords


# based on https://github.com/PacktPublishing/TensorFlow-Machine-Learning-Cookbook/blob/master/Chapter%2007/doc2vec.py

data_folder_name = '/home/chris/data/CORE/interim'
if not os.path.exists(data_folder_name):
    os.makedirs(data_folder_name)

# Start a graph session
sess = tf.Session()

# Declare model parameters
batch_size = 500
vocabulary_size = 10000
generations = 100000
model_learning_rate = 0.001

embedding_size = 300   # Word embedding size
doc_embedding_size = 300   # Document embedding size
concatenated_size = embedding_size + doc_embedding_size

num_sampled = int(batch_size/2)    # Number of negative examples to sample.
window_size = 5  # How many words to consider to the left.

# Add checkpoints to training
save_embeddings_every = 50000
print_valid_every = 50000
print_loss_every = 10000

# Declare stop words
stops = stopwords.words('english')
print('Creating Dictionary')
with open(os.path.join(data_folder_name, "subset_en")) as infile:
    docs = (doc for doc in infile if len(doc.split()) > window_size)
    word2id = helpers.build_vocab(docs, vocabulary_size, stops)
    id2word = dict(zip(word2id.values(), word2id.keys()))

valid_words = ['global', 'warming', 'climate', 'change', 'science', 'fiction']


print('Converting docs to numbers')
with open(os.path.join(data_folder_name, "subset_en")) as infile:
    docs = [helpers.doc2ids(doc, word2id) for doc in infile]
docs = [doc for doc in docs if len(doc) > window_size]
# Get validation word keys
valid_examples = [word2id[x] for x in valid_words if x in word2id]


print('Creating Model')
# Define Embeddings:
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size],
                                           -1.0, 1.0))
doc_embeddings = tf.Variable(tf.random_uniform([len(docs), doc_embedding_size],
                                               -1.0, 1.0))


# NCE loss parameters
nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size,
                                               concatenated_size],
                                              stddev=1.0 /
                                              np.sqrt(concatenated_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))


# Create data/target placeholders
x_inputs = tf.placeholder(tf.int32, shape=[None, window_size + 1])  # plus 1 for doc index
y_target = tf.placeholder(tf.float32, shape=[None, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)


# Lookup the word embedding
# Add together element embeddings in window:
embed = tf.zeros([batch_size, embedding_size])
for element in range(window_size):
    embed += tf.nn.embedding_lookup(embeddings, x_inputs[:, element])

doc_indices = tf.slice(x_inputs, [0, window_size], [batch_size, 1])
doc_embed = tf.nn.embedding_lookup(doc_embeddings, doc_indices)

# concatenate embeddings
final_embed = tf.concat([embed, tf.squeeze(doc_embed)], 1)

# Get loss from prediction
loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases,
                                     y_target, final_embed,
                                     num_sampled, vocabulary_size))

# Create optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=model_learning_rate)
train_step = optimizer.minimize(loss)

# Cosine similarity between words
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

# Create model saving operation
saver = tf.train.Saver({"embeddings": embeddings, "doc_embeddings": doc_embeddings})

# Add variable initializer.
init = tf.initialize_all_variables()
sess.run(init)

# Run the skip gram model.
print('Starting Training')
loss_vec = []
loss_x_vec = []

for i in range(generations):
    batch_inputs, batch_labels = helpers.generate_batch_data(
                                                docs, batch_size,
                                                window_size)
    feed_dict = {x_inputs: batch_inputs, y_target: batch_labels}

    # Run the train step
    sess.run(train_step, feed_dict=feed_dict)

    # Return the loss
    if (i+1) % print_loss_every == 0:
        loss_val = sess.run(loss, feed_dict=feed_dict)
        loss_vec.append(loss_val)
        loss_x_vec.append(i+1)
        print('Loss at step {} : {}'.format(i+1, loss_val))

    # Validation: Print some random words and top 5 related words
    if (i+1) % print_valid_every == 0:
        sim = sess.run(similarity, feed_dict=feed_dict)
        for j in range(len(valid_words)):
            valid_word = id2word[valid_examples[j]]
            top_k = 5  # number of nearest neighbors
            nearest = (-sim[j, :]).argsort()[1:top_k+1]
            log_str = "Nearest to {}:".format(valid_word)
            for k in range(top_k):
                close_word = id2word[nearest[k]]
                log_str = '{} {},'.format(log_str, close_word)
            print(log_str)

    # Save dictionary + embeddings
    if (i+1) % save_embeddings_every == 0:
        # Save vocabulary dictionary
        with open(os.path.join(data_folder_name, 'core_vocab.pkl'), 'wb') as f:
            pickle.dump(word2id, f)

        # Save embeddings
        model_checkpoint_path = os.path.join(os.getcwd(),
                                             data_folder_name,
                                             'core_embeddings.ckpt')
        save_path = saver.save(sess, model_checkpoint_path)
        print('Model saved in file: {}'.format(save_path))
