"""
Interactive Image Classifier & CLI Inference for CIFAR-10.
Loads the trained PyTorch CNN, renders terminal ASCII previews of images,
and outputs class probabilities and classification predictions.
"""

import os
import sys
import random
import argparse
import torch
from torchvision import transforms, datasets
from PIL import Image

# Import model definition from local package
sys.path.append(os.path.dirname(__file__))
from model import CIFAR10CNN, CIFAR10_CLASSES, CLASS_EMOJIS, get_device

ASCII_RAMP = " .:-=+*#%@"


def rgb_to_ascii(img: Image.Image, width: int = 32) -> str:
    """Converts 32x32 image to ASCII art."""
    img_gray = img.convert("L").resize((width, int(width / 2)))
    lines = []
    for y in range(img_gray.size[1]):
        row_str = ""
        for x in range(img_gray.size[0]):
            val = img_gray.getpixel((x, y))
            row_str += ASCII_RAMP[int((val / 255) * (len(ASCII_RAMP) - 1))]
        lines.append("    " + row_str)
    return "\n".join(lines)


def format_probability_bar(prob: float, width: int = 24) -> str:
    filled = int(round(prob * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"|{bar}| {prob * 100:>5.1f}%"


def load_model(checkpoint_path: str = None):
    if checkpoint_path is None:
        checkpoint_path = os.path.join(os.path.dirname(__file__), "models", "cifar10_cnn.pt")

    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint not found at '{checkpoint_path}'.")
        print("Please train the model first by running:\n  python3 08_image_classification/train.py")
        sys.exit(1)

    device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = CIFAR10CNN(num_classes=10).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, checkpoint.get("classes", CIFAR10_CLASSES), checkpoint.get("best_acc", 0.0), device


def predict_image(model, img: Image.Image, classes: list, device: torch.device, true_label: str = None):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = model.predict_proba(tensor).cpu().squeeze().numpy()

    pred_idx = int(probs.argmax())
    pred_class = classes[pred_idx]
    pred_emoji = CLASS_EMOJIS.get(pred_class, "")

    print("\n" + "=" * 68)
    print(f" 🖼️ CLASSIFICATION VERDICT: {pred_class.upper()} {pred_emoji}")
    print("=" * 68)

    print("\n  Terminal Image Preview (32x32 pixels):")
    print(rgb_to_ascii(img, width=32))
    print()

    if true_label is not None:
        is_correct = pred_class.lower() == true_label.lower()
        status = "✅ CORRECT MATCH!" if is_correct else "❌ MISCLASSIFIED"
        print(f"  • Ground Truth Label : {true_label.upper()} {CLASS_EMOJIS.get(true_label, '')}")
        print(f"  • Predicted Label    : {pred_class.upper()} {pred_emoji} ({status})")
    else:
        print(f"  • Predicted Label    : {pred_class.upper()} {pred_emoji}")

    print(f"  • Top Match Confidence: {probs[pred_idx] * 100:.2f}%\n")

    print("  Top 4 Class Probabilities:")
    top_indices = probs.argsort()[::-1][:4]
    for idx in top_indices:
        c_name = classes[idx]
        emoji = CLASS_EMOJIS.get(c_name, "")
        bar = format_probability_bar(probs[idx])
        marker = " 👈 (Top Match)" if idx == pred_idx else ""
        print(f"    {emoji} {c_name:<11} {bar}{marker}")

    print("=" * 68 + "\n")
    return pred_class


def run_interactive(model, classes, test_dir, device):
    test_dataset = datasets.ImageFolder(test_dir)

    while True:
        print("\n" + "=" * 60)
        print(" 🖼️ CIFAR-10 IMAGE CLASSIFIER - INTERACTIVE MENU 🖼️")
        print("=" * 60)
        print("  1) Classify a Random Test Image")
        print("  2) Select a Specific Category to Test (e.g. Airplane, Cat)")
        print("  3) Batch Test 10 Random Samples (Accuracy Check)")
        print("  4) Exit")
        print("-" * 60)

        choice = input("Enter choice (1-4): ").strip()

        if choice == "1":
            rand_idx = random.randint(0, len(test_dataset) - 1)
            img, label_idx = test_dataset[rand_idx]
            true_label = classes[label_idx]
            print(f"\nSelected Test Sample #{rand_idx}")
            predict_image(model, img, classes, device, true_label=true_label)

        elif choice == "2":
            print("\nAvailable Categories:")
            for idx, c in enumerate(classes, 1):
                print(f"  {idx:>2}) {CLASS_EMOJIS.get(c, '')} {c.title()}")

            cat_choice = input(f"\nSelect category (1-{len(classes)}): ").strip()
            if cat_choice.isdigit() and 1 <= int(cat_choice) <= len(classes):
                selected_class = classes[int(cat_choice) - 1]
                cat_folder = os.path.join(test_dir, selected_class)
                files = os.listdir(cat_folder)
                chosen_file = random.choice(files)
                img = Image.open(os.path.join(cat_folder, chosen_file))
                predict_image(model, img, classes, device, true_label=selected_class)
            else:
                print("Invalid selection.")

        elif choice == "3":
            print("\nEvaluating 10 Random Test Images:")
            print(f"{'#':<3} | {'Sample ID':<10} | {'True Label':<16} | {'Predicted Label':<16} | {'Verdict'}")
            print("-" * 65)
            correct_cnt = 0
            for i in range(1, 11):
                rand_idx = random.randint(0, len(test_dataset) - 1)
                img, label_idx = test_dataset[rand_idx]
                true_label = classes[label_idx]

                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
                ])
                tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred_idx = int(model.predict_proba(tensor).argmax())
                pred_label = classes[pred_idx]
                match = pred_label == true_label
                if match:
                    correct_cnt += 1
                v_str = "✅ Correct" if match else "❌ Miss"
                print(f"{i:<3} | #{rand_idx:<9} | {true_label:<16} | {pred_label:<16} | {v_str}")
            print(f"\nBatch Score: {correct_cnt} / 10 correct ({correct_cnt * 10}%)\n")

        elif choice == "4":
            print("Exiting Image Classifier. Happy Deep Learning!\n")
            break
        else:
            print("Invalid option. Please choose 1 to 4.")


def main():
    parser = argparse.ArgumentParser(description="Predict image category with CIFAR-10 CNN.")
    parser.add_argument("--image-path", type=str, help="Path to local image file to classify")
    parser.add_argument("--sample-index", type=int, help="Index of CIFAR-10 test sample (0-9999)")

    args = parser.parse_args()

    model, classes, best_acc, device = load_model()
    test_dir = os.path.join(os.path.dirname(__file__), "data", "cifar10", "test")

    if args.image_path:
        if not os.path.exists(args.image_path):
            print(f"Error: Image file not found at '{args.image_path}'.")
            return
        img = Image.open(args.image_path).convert("RGB")
        predict_image(model, img, classes, device)
    elif args.sample_index is not None:
        test_dataset = datasets.ImageFolder(test_dir)
        idx = max(0, min(args.sample_index, len(test_dataset) - 1))
        img, label_idx = test_dataset[idx]
        predict_image(model, img, classes, device, true_label=classes[label_idx])
    else:
        run_interactive(model, classes, test_dir, device)


if __name__ == "__main__":
    main()
