import json
import numpy as np
import onnxruntime as ort
import argparse


def preprocess_text(text, tokenizer_file):
    with open(tokenizer_file, 'r') as f:
        tokenizer = json.load(f)

    oov_token = '<OOV>'
    words = text.lower().split()
    sequence = [tokenizer.get(word, tokenizer.get(oov_token, 1)) for word in words]
    sequence = sequence[:30]  # Truncate to max_len
    padded = np.zeros(30, dtype=np.int32)
    padded[:len(sequence)] = sequence  # Pad with zeros
    return padded


def main():
    parser = argparse.ArgumentParser(description='Classify news text using ONNX model')
    parser.add_argument('text', type=str, help='Text to classify')
    args = parser.parse_args()

    vector = preprocess_text(args.text, 'models/news_classifier/vocab.json')

    session = ort.InferenceSession('models/news_classifier/model.onnx')
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_data = vector.reshape(1, 30)
    outputs = session.run([output_name], {input_name: input_data})

    with open('models/news_classifier/scaler.json', 'r') as f:
        label_map = json.load(f)

    probabilities = outputs[0][0]
    predicted_idx = np.argmax(probabilities)
    label = label_map[str(predicted_idx)]
    score = probabilities[predicted_idx]
    print(f'Classification: {label} (Score: {score:.4f})')


if __name__ == "__main__":
    main()