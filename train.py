import json
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
import joblib     #for store tokenizer file
import config
import tensorflow as tf

class VideoDescriptionTrain:
    def _init_(self, config):
        self.train_path = config.train_path
        self.test_path = config.test_path
        self.max_length = config.max_length
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.num_decoder_tokens = config.num_decoder_tokens
        self.x_data = {}

        # processed data
        self.tokenizer = None

        # models
        self.model = None
        self.save_model_path = config.save_model_path

    def preprocessing(self):
        print("Preprocessing data...")
        TRAIN_LABEL_PATH = os.path.join(self.train_path, 'training_label.json')
        with open(TRAIN_LABEL_PATH) as data_file:   #open and read the json file
            y_data = json.load(data_file)
        #empty dictionationary for training data and vocabulary
        train_list = []
        vocab_list = []
        for y in y_data:  # Load entire dataset
            for caption in y['caption']:
                caption = "<bos> " + caption + " <eos>"
                if 6 <= len(caption.split()) <= 10:  # Filter captions with length between 6 and 10
                    train_list.append([caption, y['id']])
                    vocab_list.append(caption)

        self.tokenizer = Tokenizer(num_words=self.num_decoder_tokens)
        self.tokenizer.fit_on_texts(vocab_list)

        TRAIN_FEATURE_DIR = os.path.join(self.train_path, 'feat')
        for filename in os.listdir(TRAIN_FEATURE_DIR):
            self.x_data[filename[:-4]] = os.path.join(TRAIN_FEATURE_DIR, filename)  #add each feature into the x_data with his id
        print("Data preprocessing completed.")
        return train_list

    def load_chunk(self, filename):
        f = np.load(filename, allow_pickle=True)
        # Assuming the loaded chunk is of shape (time_steps_encoder, num_encoder_tokens)
        # Reshape it to match the expected shape (None, None, num_encoder_tokens)
        return np.resize(f, (10, 4096))

    def load_dataset(self, training_list):
        print("Loading dataset...")

        def generator():
            for i in range(0, min(100, len(training_list)), self.batch_size):
                batch_data = training_list[i:i + self.batch_size]
                encoder_input_data = []
                decoder_input_data = []
                decoder_target_data = []
                for caption, videoId in batch_data:
                    encoder_input = self.load_chunk(self.x_data[videoId])
                    train_sequence = self.tokenizer.texts_to_sequences([caption])[0]
                    train_sequence = pad_sequences([train_sequence], padding='post', truncating='post',
                                                   maxlen=self.max_length)[0]
                    decoder_input = to_categorical(train_sequence, self.num_decoder_tokens)[:-1] #convert into the one-hot encoder
                    decoder_target = to_categorical(train_sequence, self.num_decoder_tokens)[1:]
                    encoder_input_data.append(encoder_input)
                    decoder_input_data.append(decoder_input)
                    decoder_target_data.append(decoder_target)
                yield ((np.array(encoder_input_data), np.array(decoder_input_data)), np.array(decoder_target_data)) #return the value while maintaining the currrent value
#here in shape batch size,seq_length,number of tokens
#first for encoder input
#second for decoder input
#third for decoder output
#this is for each batch of generator
        output_signature = (
            ((tf.TensorSpec(shape=(None, 10, 4096), dtype=tf.float32),
              tf.TensorSpec(shape=(None, None, self.num_decoder_tokens), dtype=tf.float32)),
             tf.TensorSpec(shape=(None, None, self.num_decoder_tokens), dtype=tf.float32))
        )

        dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature) #create tensorflow dataset
        return dataset

    def build_model(self):
        print("Building model...")
        encoder_inputs = Input(shape=(10, 4096), name="encoder_inputs")
        encoder = LSTM(config.latent_dim, return_state=True, return_sequences=True, name='encoder_lstm')
        _, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]  # hidden state and cell state of last lstm layers

        decoder_inputs = Input(shape=(None, self.num_decoder_tokens), name="decoder_inputs")
        decoder_lstm = LSTM(config.latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax', name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_outputs)

        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # Compile the model with the loss function
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("Model built.")

    def train_model(self):
        print("Training model...")
        training_list = self.preprocessing()
        train_dataset = self.load_dataset(training_list)
        steps_per_epoch = len(training_list) // self.batch_size

        optimizer = tf.keras.optimizers.Adam()  # Define optimizer

        early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='min')

        for epoch in range(self.epochs):
            print("Epoch {}/{}".format(epoch + 1, self.epochs))

            total_loss = 0
            total_accuracy = 0
            num_batches = 0

            for step, ((encoder_input_data, decoder_input_data), decoder_target_data) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    decoder_outputs = self.model([encoder_input_data, decoder_input_data], training=True)
                    loss = tf.keras.losses.categorical_crossentropy(decoder_target_data, decoder_outputs)

                gradients = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                # Compute mean loss and accuracy
                batch_loss = np.mean(loss.numpy())
                batch_accuracy = np.mean(tf.keras.metrics.categorical_accuracy(decoder_target_data, decoder_outputs))

                total_loss += batch_loss
                total_accuracy += batch_accuracyimport json
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
import joblib     #for store tokenizer file
import config
import tensorflow as tf

class VideoDescriptionTrain:
    def _init_(self, config):
        self.train_path = config.train_path
        self.test_path = config.test_path
        self.max_length = config.max_length
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.num_decoder_tokens = config.num_decoder_tokens
        self.x_data = {}

        # processed data
        self.tokenizer = None

        # models
        self.model = None
        self.save_model_path = config.save_model_path

    def preprocessing(self):
        print("Preprocessing data...")
        TRAIN_LABEL_PATH = os.path.join(self.train_path, 'training_label.json')
        with open(TRAIN_LABEL_PATH) as data_file:   #open and read the json file
            y_data = json.load(data_file)
        #empty dictionationary for training data and vocabulary
        train_list = []
        vocab_list = []
        for y in y_data:  # Load entire dataset
            for caption in y['caption']:
                caption = "<bos> " + caption + " <eos>"
                if 6 <= len(caption.split()) <= 10:  # Filter captions with length between 6 and 10
                    train_list.append([caption, y['id']])
                    vocab_list.append(caption)

        self.tokenizer = Tokenizer(num_words=self.num_decoder_tokens)
        self.tokenizer.fit_on_texts(vocab_list)

        TRAIN_FEATURE_DIR = os.path.join(self.train_path, 'feat')
        for filename in os.listdir(TRAIN_FEATURE_DIR):
            self.x_data[filename[:-4]] = os.path.join(TRAIN_FEATURE_DIR, filename)  #add each feature into the x_data with his id
        print("Data preprocessing completed.")
        return train_list

    def load_chunk(self, filename):
        f = np.load(filename, allow_pickle=True)
        # Assuming the loaded chunk is of shape (time_steps_encoder, num_encoder_tokens)
        # Reshape it to match the expected shape (None, None, num_encoder_tokens)
        return np.resize(f, (10, 4096))

    def load_dataset(self, training_list):
        print("Loading dataset...")

        def generator():
            for i in range(0, min(100, len(training_list)), self.batch_size):
                batch_data = training_list[i:i + self.batch_size]
                encoder_input_data = []
                decoder_input_data = []
                decoder_target_data = []
                for caption, videoId in batch_data:
                    encoder_input = self.load_chunk(self.x_data[videoId])
                    train_sequence = self.tokenizer.texts_to_sequences([caption])[0]
                    train_sequence = pad_sequences([train_sequence], padding='post', truncating='post',
                                                   maxlen=self.max_length)[0]
                    decoder_input = to_categorical(train_sequence, self.num_decoder_tokens)[:-1] #convert into the one-hot encoder
                    decoder_target = to_categorical(train_sequence, self.num_decoder_tokens)[1:]
                    encoder_input_data.append(encoder_input)
                    decoder_input_data.append(decoder_input)
                    decoder_target_data.append(decoder_target)
                yield ((np.array(encoder_input_data), np.array(decoder_input_data)), np.array(decoder_target_data)) #return the value while maintaining the currrent value
#here in shape batch size,seq_length,number of tokens
#first for encoder input
#second for decoder input
#third for decoder output
#this is for each batch of generator
        output_signature = (
            ((tf.TensorSpec(shape=(None, 10, 4096), dtype=tf.float32),
              tf.TensorSpec(shape=(None, None, self.num_decoder_tokens), dtype=tf.float32)),
             tf.TensorSpec(shape=(None, None, self.num_decoder_tokens), dtype=tf.float32))
        )

        dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature) #create tensorflow dataset
        return dataset

    def build_model(self):
        print("Building model...")
        encoder_inputs = Input(shape=(10, 4096), name="encoder_inputs")
        encoder = LSTM(config.latent_dim, return_state=True, return_sequences=True, name='encoder_lstm')
        _, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]  # hidden state and cell state of last lstm layers

        decoder_inputs = Input(shape=(None, self.num_decoder_tokens), name="decoder_inputs")
        decoder_lstm = LSTM(config.latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax', name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_outputs)

        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # Compile the model with the loss function
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("Model built.")

    def train_model(self):
        print("Training model...")
        training_list = self.preprocessing()
        train_dataset = self.load_dataset(training_list)
        steps_per_epoch = len(training_list) // self.batch_size

        optimizer = tf.keras.optimizers.Adam()  # Define optimizer

        early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='min')

        for epoch in range(self.epochs):
            print("Epoch {}/{}".format(epoch + 1, self.epochs))

            total_loss = 0
            total_accuracy = 0
            num_batches = 0

            for step, ((encoder_input_data, decoder_input_data), decoder_target_data) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    decoder_outputs = self.model([encoder_input_data, decoder_input_data], training=True)
                    loss = tf.keras.losses.categorical_crossentropy(decoder_target_data, decoder_outputs)

                gradients = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                # Compute mean loss and accuracy
                batch_loss = np.mean(loss.numpy())
                batch_accuracy = np.mean(tf.keras.metrics.categorical_accuracy(decoder_target_data, decoder_outputs))

                total_loss += batch_loss
                total_accuracy += batch_accuracy
                num_batches += 1

                if step % 100 == 0:
                    print("Step {}/{} - Loss: {:.4f} - Accuracy: {:.4f}".format(step, steps_per_epoch, batch_loss,
                                                                                batch_accuracy))

            # Print average loss and accuracy for the epoch
            avg_loss = total_loss / num_batches
            avg_accuracy = total_accuracy / num_batches
            print("Epoch {}/{} - Avg Loss: {:.4f} - Avg Accuracy: {:.4f}".format(epoch + 1, self.epochs, avg_loss,
                                                                                 avg_accuracy))

            # Optionally, evaluate on validation set and apply early stopping logic
            # early_stopping.update(loss)

        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)

        self.model.save(os.path.join(self.save_model_path, 'video_captioning_model.h5'))
        with open(os.path.join(self.save_model_path, 'tokenizer.pkl'), 'wb') as file:
            joblib.dump(self.tokenizer, file)
        print("Training completed.")


if _name_ == "_main_":
    video_to_text = VideoDescriptionTrain(config)
    video_to_text.build_model()
    video_to_text.train_model()

                num_batches += 1

                if step % 100 == 0:
                    print("Step {}/{} - Loss: {:.4f} - Accuracy: {:.4f}".format(step, steps_per_epoch, batch_loss,
                                                                                batch_accuracy))

            # Print average loss and accuracy for the epoch
            avg_loss = total_loss / num_batches
            avg_accuracy = total_accuracy / num_batches
            print("Epoch {}/{} - Avg Loss: {:.4f} - Avg Accuracy: {:.4f}".format(epoch + 1, self.epochs, avg_loss,
                                                                                 avg_accuracy))

            # Optionally, evaluate on validation set and apply early stopping logic
            # early_stopping.update(loss)

        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)

        self.model.save(os.path.join(self.save_model_path, 'video_captioning_model.h5'))
        with open(os.path.join(self.save_model_path, 'tokenizer.pkl'), 'wb') as file:
            joblib.dump(self.tokenizer, file)
        print("Training completed.")


if _name_ == "_main_":
    video_to_text = VideoDescriptionTrain(config)
    video_to_text.build_model()
    video_to_text.train_model()
