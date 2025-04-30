import json
import numpy as np
import onnxruntime as ort
import argparse


def preprocess_text(text, vocab_file, scaler_file):
    with open(vocab_file, "r") as f:
        tfidf_data = json.load(f)
    vocab, idf = tfidf_data["vocab"], np.array(tfidf_data["idf"])

    with open(scaler_file, "r") as f:
        scaler_data = json.load(f)
    mean, scale = np.array(scaler_data["mean"]), np.array(scaler_data["scale"])

    vector = np.zeros(5000, dtype=np.float32)
    words = text.lower().split()
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1
    for word, count in word_counts.items():
        if word in vocab:
            vector[vocab[word]] = count * idf[vocab[word]]

    # Scale
    vector = (vector - mean) / scale
    return vector


def main():
    parser = argparse.ArgumentParser(description='Classify text using ONNX model')
    parser.add_argument('text', type=str, help='Text to classify')
    args = parser.parse_args()

    vector = preprocess_text(
        args.text,
        "models/spam_classifier/vocab.json",
        "models/spam_classifier/scaler.json"
    )

    session = ort.InferenceSession("models/spam_classifier/model.onnx")
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_data = vector.reshape(1, 5000).astype(np.float32)
    outputs = session.run([output_name], {input_name: input_data})
    print("Classification score:", outputs[0][0][0])


if __name__ == "__main__":
    main()