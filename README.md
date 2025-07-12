# miniproject5
Multiclass Fish Image Classification
Data Preprocessing and Augmentation
            Rescale images to [0, 1] range.
            Apply data augmentation techniques like rotation, zoom, and flipping to enhance model robustness.
Model Training
            Train a CNN model from scratch.
            Experiment with five pre-trained models (VGG16)
            Fine-tune the pre-trained models on the fish dataset.
            Save the trained model (max accuracy model ) in .h5 and convert into keras or .pkl format for future use.
Model Evaluation
          Compare metrics such as accuracy, precision, recall, F1-score, and confusion matrix across all models.
          Visualize training history (accuracy and loss) for each model.
Deployment
          Build a Streamlit application to:
          Allow users to upload fish images.
          Predict and display the fish category.
          Provide model confidence scores.

