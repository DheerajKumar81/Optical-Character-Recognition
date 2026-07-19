import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import difflib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress HuggingFace warnings for cleaner terminal output
warnings.filterwarnings("ignore")

def analyze_text_metrics(true_text, pred_text):
    """
    Compares the true text with the predicted text to calculate Character Error Rate (CER),
    Accuracy, and generates character mappings for a confusion matrix.
    """
    true_text = true_text.lower()
    pred_text = pred_text.lower()
    
    # Use SequenceMatcher to find the exact character edits between true and predicted
    matcher = difflib.SequenceMatcher(None, true_text, pred_text)
    
    substitutions, deletions, insertions, correct = 0, 0, 0, 0
    char_mapping = [] # List of (true_char, pred_char)
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for c in true_text[i1:i2]:
                char_mapping.append((c, c))
                correct += 1
        elif tag == 'replace':
            t_sub = true_text[i1:i2]
            p_sub = pred_text[j1:j2]
            max_len = max(len(t_sub), len(p_sub))
            for k in range(max_len):
                t_c = t_sub[k] if k < len(t_sub) else '-' # '-' denotes missing character
                p_c = p_sub[k] if k < len(p_sub) else '-'
                char_mapping.append((t_c, p_c))
                if t_c == '-': insertions += 1
                elif p_c == '-': deletions += 1
                else: substitutions += 1
        elif tag == 'delete':
            for c in true_text[i1:i2]:
                char_mapping.append((c, '-'))
                deletions += 1
        elif tag == 'insert':
            for c in pred_text[j1:j2]:
                char_mapping.append(('-', c))
                insertions += 1
                
    total_chars = max(len(true_text), 1)
    
    # Calculate Character Error Rate (CER) and Accuracy
    cer = (substitutions + deletions + insertions) / total_chars
    accuracy = max(0, 1 - cer) * 100
    
    return accuracy, cer, char_mapping

def plot_dashboard(image, true_text, pred_text, accuracy, cer, char_mapping):
    """
    Displays the image, metrics, and a character-level confusion matrix.
    """
    # 1. Prepare data for the confusion matrix
    unique_true = sorted(list(set([pair[0] for pair in char_mapping])))
    unique_pred = sorted(list(set([pair[1] for pair in char_mapping])))
    
    matrix = np.zeros((len(unique_true), len(unique_pred)), dtype=int)
    
    for t_c, p_c in char_mapping:
        row = unique_true.index(t_c)
        col = unique_pred.index(p_c)
        matrix[row, col] += 1

    # 2. Setup the visual dashboard
    fig = plt.figure(figsize=(12, 6))
    fig.canvas.manager.set_window_title("Handwritten Text OCR Results")
    
    # Left side: Image and Metrics
    ax_img = plt.subplot(1, 2, 1)
    ax_img.imshow(image)
    ax_img.axis('off')
    
    metrics_text = (
        f"Ground Truth: '{true_text}'\n"
        f"Prediction: '{pred_text}'\n\n"
        f"Character Accuracy: {accuracy:.1f}%\n"
        f"Character Error Rate: {cer:.3f}"
    )
    
    color = "green" if true_text.lower() == pred_text.lower() else "red"
    ax_img.set_title(metrics_text, loc='left', color=color, fontsize=12, pad=15)

    # Right side: Confusion Matrix
    ax_matrix = plt.subplot(1, 2, 2)
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, 
                xticklabels=unique_pred, yticklabels=unique_true, ax=ax_matrix)
    ax_matrix.set_xlabel("Predicted Characters")
    ax_matrix.set_ylabel("Actual Characters")
    ax_matrix.set_title("Character-Level Confusion Matrix")

    plt.tight_layout()
    plt.show()

def main():
    # =========================================================
    # USER INPUTS: Change these to test your own images!
    # =========================================================
    image_path = "Deep_learning_approach\messi2.0.png" # PATH TO YOUR IMAGE
    ground_truth_text = "MESSI"         # WHAT THE IMAGE ACTUALLY SAYS
    # =========================================================

    print("Loading Microsoft TrOCR Handwritten Model (This may take a minute on first run)...")
    try:
        processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
        model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    except Exception as e:
        print("Error loading model. Ensure you have an internet connection.")
        return

    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"\n❌ Error: Could not find image at '{image_path}'")
        print("Please place an image file there and update the 'image_path' variable.")
        return

    print("\nProcessing image and predicting text...")
    # Preprocess image and generate text
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Calculate performance metrics
    print("\nCalculating metrics...")
    accuracy, cer, char_mapping = analyze_text_metrics(ground_truth_text, predicted_text)

    # Output to terminal
    print("-" * 40)
    print(f"Expected Text  : {ground_truth_text}")
    print(f"Predicted Text : {predicted_text}")
    print(f"Accuracy       : {accuracy:.2f}%")
    print(f"CER (Error)    : {cer:.3f}")
    print("-" * 40)

    # Open visual dashboard
    plot_dashboard(image, ground_truth_text, predicted_text, accuracy, cer, char_mapping)

if __name__ == "__main__":
    main()