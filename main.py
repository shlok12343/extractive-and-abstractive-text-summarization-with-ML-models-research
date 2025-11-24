from src.data_loader import get_training_data
from src.preprocessor import TextPreprocessor
from src.model import SummaryModel
from src.summarizer import Summarizer
import os

def load_single_lecture_note(file_path):
    """Loads a single lecture note for summarization."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return ""

def main():
    """
    Main function to run the summarization pipeline.
    """
    # 1. Load and prepare training data
    # This will now use the CNN/Daily Mail dataset by default
    print("Loading training data...")
    sentences, labels = get_training_data(use_cnn_dailymail=True, sample_size=200)

    if not sentences:
        print("No training data found. Exiting.")
        return

    # 2. Preprocess the training data
    print("Preprocessing data...")
    preprocessor = TextPreprocessor()
    X_train = preprocessor.fit_transform(sentences)

    # 3. Train the model
    print("Training model...")
    summary_model = SummaryModel()
    summary_model.train(X_train, labels)
    print("Model training complete.")

    # 4. Create a summarizer
    summarizer = Summarizer(model=summary_model, preprocessor=preprocessor)

    # 5. Load a lecture note to be summarized
    # We'll summarize one of our local files for demonstration
    lecture_file_to_summarize = "data/lecture_notes_rl.txt"
    print(f"\nLoading '{lecture_file_to_summarize}' for summarization...")
    lecture_to_summarize = load_single_lecture_note(lecture_file_to_summarize)

    if not lecture_to_summarize:
        print("No lecture notes to summarize. Exiting.")
        return

    # 6. Generate and print the summary
    print("Generating summary...")
    summary = summarizer.summarize(lecture_to_summarize, num_sentences=5)

    print("\n--- Original Text ---")
    print(lecture_to_summarize)
    print("\n--- Summary ---")
    print(summary)


if __name__ == "__main__":
    main()
