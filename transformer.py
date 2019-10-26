from keras.models import model_from_json
from keras_transformer import get_model, decode, get_custom_objects
import numpy as np
import os

EPOCHS = 0
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.1

# Expand small sets (OPTIONAL = 1)
DATA_MULTIPLIER = 1024

EMBED_DIM = 32
LAYERS = 2
ATTN_HEADS = 8
HIDDEN_DIM = 64
DROPOUT = 0.1

TOP_K = 5
BEAM_TEMP = 0.1

MODEL_NAME = 'models/trained/transformer'


def build_token_dict(token_list):
    token_dict = {
        '<PAD>': 0,
        '<START>': 1,
        '<END>': 2,
    }

    for tokens in token_list:
        for token in tokens:
            if token not in token_dict:
                token_dict[token] = len(token_dict)

    return token_dict


def text_as_tokens(text):
    return text.split(' ')


if __name__ == "__main__":
    # Data_X
    source_tokens = [
        text_as_tokens("este é o primeiro livro que eu fiz ."),
        text_as_tokens(
            "vou então muito rapidamente partilhar convosco algumas histórias de algumas coisas mágicas que aconteceram ."
        )
    ]
    # Data_y
    target_tokens = [
        text_as_tokens("this is the first book i 've ever done ."),
        text_as_tokens(
            "so i 'll just share with you some stories very quickly of some magical things that have happened ."
        )
    ]

    source_token_dict = build_token_dict(source_tokens)
    target_token_dict = build_token_dict(target_tokens)
    target_token_dict_inv = {v: k for k, v in target_token_dict.items()}

    # Add special tokens
    encode_tokens = [['<START>'] + tokens + ['<END>']
                     for tokens in source_tokens]
    decode_tokens = [['<START>'] + tokens + ['<END>']
                     for tokens in target_tokens]
    output_tokens = [tokens + ['<END>', '<PAD>'] for tokens in target_tokens]

    # Padding
    source_max_len = max(map(len, encode_tokens))
    target_max_len = max(map(len, decode_tokens))

    encode_tokens = [tokens + ['<PAD>'] *
                     (source_max_len - len(tokens)) for tokens in encode_tokens]
    decode_tokens = [tokens + ['<PAD>'] *
                     (target_max_len - len(tokens)) for tokens in decode_tokens]
    output_tokens = [tokens + ['<PAD>'] *
                     (target_max_len - len(tokens)) for tokens in output_tokens]

    encode_input = [list(map(lambda x: source_token_dict[x], tokens))
                    for tokens in encode_tokens]
    decode_input = [list(map(lambda x: target_token_dict[x], tokens))
                    for tokens in decode_tokens]
    decode_output = [list(map(lambda x: [target_token_dict[x]], tokens))
                     for tokens in output_tokens]

    if EPOCHS > 0:
        # Build & fit model
        model = get_model(
            token_num=max(len(source_token_dict), len(target_token_dict)),
            embed_dim=EMBED_DIM,
            encoder_num=LAYERS,
            decoder_num=LAYERS,
            head_num=ATTN_HEADS,
            hidden_dim=HIDDEN_DIM,
            dropout_rate=DROPOUT,
            use_same_embed=False,  # Use different embeddings for different languages
        )

        model.compile('adam', 'sparse_categorical_crossentropy')
        model.summary()

        model.fit(
            x=[np.array(encode_input * DATA_MULTIPLIER),
               np.array(decode_input * DATA_MULTIPLIER)],
            y=np.array(decode_output * DATA_MULTIPLIER),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT
        )

        model.save_weights(os.path.join(MODEL_NAME, 'model_weights.h5'))

        # Save the model architecture
        with open(os.path.join(MODEL_NAME, 'model.json'), 'w') as fh:
            fh.write(model.to_json())
    else:
        # Model reconstruction from JSON file
        with open(os.path.join(MODEL_NAME, 'model.json'), 'r') as fh:
            model = model_from_json(fh.read(), get_custom_objects())

        # Load weights into the new model
        model.load_weights(os.path.join(MODEL_NAME, 'model_weights.h5'))

        model.compile('adam', 'sparse_categorical_crossentropy')
        model.summary()

    # Predict with beam search
    decoded = decode(
        model,
        encode_input,  # The test set
        start_token=target_token_dict['<START>'],
        end_token=target_token_dict['<END>'],
        pad_token=target_token_dict['<PAD>'],
        top_k=TOP_K,
        temperature=BEAM_TEMP,
    )

    def predict(text):
        return ' '.join(map(lambda x: target_token_dict_inv[x], text))

    # Predict the first two examples in data
    print("este é o primeiro livro que eu fiz .")
    print(predict(decoded[0][1:-1]))
    print("\nvou então muito rapidamente partilhar convosco algumas histórias de algumas coisas mágicas que aconteceram .")
    print(predict(decoded[1][1:-1]))
