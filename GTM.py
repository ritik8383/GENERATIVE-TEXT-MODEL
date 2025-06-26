import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import random
import sys

# --- Configuration ---
# Path to the text file you want to train the model on
# You can replace this with your own text file.
# For demonstration, a simple text will be used if no file is found.
TEXT_FILE_PATH = 'input_text.txt'

# Length of the sequence (number of characters) the model will learn from
SEQUENCE_LENGTH = 100

# Step size to move through the text when creating sequences
STEP_SIZE = 3

# Number of training epochs
EPOCHS = 20 # You can increase this for better quality, but it will take longer

# Batch size for training
BATCH_SIZE = 128

# Directory to save model checkpoints
CHECKPOINT_DIR = './checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- Data Preparation ---

def load_and_prepare_text(file_path):
    """
    Loads text from a file, preprocesses it, and creates character mappings.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().lower() # Convert to lowercase for consistent character set
    except FileNotFoundError:
        print(f"Warning: '{file_path}' not found. Using a default sample text for demonstration.")
        text = """
        The quick brown fox jumps over the lazy dog. This is a classic pangram.
        Neural networks are a set of algorithms, inspired by the functioning of the human brain,
        that are designed to recognize patterns. They interpret sensory data through a kind of
        machine perception, labeling or clustering raw input. The patterns they recognize
        are numerical, contained in vectors, into which all real-world data, be it images,
        sound, text, or time series, must be translated.
        LSTMs are very powerful in time series prediction because of their ability to
        store long-term dependencies. They are a special kind of recurrent neural network.
        """
        # Save the default text to a file for future runs
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)

    # Get all unique characters in the text, sort them, and create mappings
    chars = sorted(list(set(text)))
    char_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_char = {i: ch for i, ch in enumerate(chars)}
    num_chars = len(chars)

    print(f"Total characters: {len(text)}")
    print(f"Unique characters: {num_chars}")

    # Create input sequences and corresponding next characters
    data_x = []
    data_y = []
    for i in range(0, len(text) - SEQUENCE_LENGTH, STEP_SIZE):
        seq_in = text[i:i + SEQUENCE_LENGTH]
        seq_out = text[i + SEQUENCE_LENGTH]
        data_x.append([char_to_int[char] for char in seq_in])
        data_y.append(char_to_int[seq_out])

    num_patterns = len(data_x)
    print(f"Total patterns: {num_patterns}")

    # Reshape input to be [samples, time steps, features]
    X = np.reshape(data_x, (num_patterns, SEQUENCE_LENGTH, 1))
    # Normalize input to range 0-1
    X = X / float(num_chars)

    # One-hot encode output variable
    y = tf.keras.utils.to_categorical(data_y, num_classes=num_chars)

    return X, y, char_to_int, int_to_char, num_chars, text


# --- Model Definition ---

def build_model(sequence_length, num_chars):
    """
    Builds the LSTM neural network model.
    """
    model = Sequential([
        LSTM(256, input_shape=(sequence_length, 1), return_sequences=True),
        Dropout(0.2),
        LSTM(256),
        Dropout(0.2),
        Dense(num_chars, activation='softmax')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model

# --- Text Generation ---

def generate_text(model, num_chars_to_generate, start_sequence, char_to_int, int_to_char, num_chars, sequence_length):
    """
    Generates text using the trained model.
    """
    if not start_sequence:
        print("Error: Start sequence cannot be empty.")
        return ""

    # Ensure start_sequence is at least SEQUENCE_LENGTH long
    if len(start_sequence) < sequence_length:
        print(f"Warning: Start sequence too short. Padding with ' ' to length {sequence_length}.")
        start_sequence = start_sequence.ljust(sequence_length).lower()
    else:
        start_sequence = start_sequence[-sequence_length:].lower() # Take the last part if too long

    generated_text = start_sequence
    pattern = [char_to_int[char] for char in start_sequence]

    sys.stdout.write(generated_text) # Print the initial sequence

    for _ in range(num_chars_to_generate):
        x = np.reshape(pattern, (1, sequence_length, 1))
        x = x / float(num_chars)
        prediction = model.predict(x, verbose=0)[0]
        
        # Sample the next character from the probability distribution
        # Using np.argmax gives the most probable, but np.random.choice adds creativity
        index = np.random.choice(len(prediction), p=prediction) # Sample based on probabilities
        
        result = int_to_char[index]
        generated_text += result
        sys.stdout.write(result)

        # Update pattern for the next prediction
        pattern.append(index)
        pattern = pattern[1:len(pattern)] # Remove the first character, add the new one

    sys.stdout.write('\n')
    return generated_text

# --- Main Execution ---

if __name__ == '__main__':
    print("--- Preparing Data ---")
    X, y, char_to_int, int_to_char, num_chars, raw_text = load_and_prepare_text(TEXT_FILE_PATH)

    print("\n--- Building Model ---")
    model = build_model(SEQUENCE_LENGTH, num_chars)
    model.summary()

    # Define callbacks for saving best model and early stopping
    filepath = os.path.join(CHECKPOINT_DIR, "weights-improvement-{epoch:02d}-{loss:.4f}.keras")
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    
    # Early stopping to prevent overfitting if loss doesn't improve
    early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True, verbose=1)
    
    callbacks_list = [checkpoint, early_stopping]

    # Load existing weights if available to resume training or generate
    latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if latest_checkpoint:
        print(f"\n--- Loading weights from {latest_checkpoint} ---")
        model.load_weights(latest_checkpoint)
        print("Model weights loaded.")
    else:
        print("\n--- No existing weights found. Training new model ---")
        # Train the model
        model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks_list)
        print("\nTraining complete.")

    # --- Generate Text ---
    print("\n--- Generating Text ---")

    # Pick a random starting point from the input text
    start_index = np.random.randint(0, len(raw_text) - SEQUENCE_LENGTH)
    random_start_sequence = raw_text[start_index : start_index + SEQUENCE_LENGTH]
    
    print(f"\nSeed for generation (first {SEQUENCE_LENGTH} chars):")
    print("--------------------------------------------------")
    print(random_start_sequence)
    print("--------------------------------------------------")

    generated_paragraph = generate_text(model, 500, random_start_sequence, char_to_int, int_to_char, num_chars, SEQUENCE_LENGTH)
    
    print("\n\n--- Generated Paragraph ---")
    print(generated_paragraph)
    print("---------------------------")
    print("\nModel trained and text generated!")

