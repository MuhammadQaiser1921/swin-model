def combine_predictions(video_prediction, audio_prediction):
    """
    Combines predictions from video and audio models using a logical OR approach.
    
    Args:
        video_prediction (int): Prediction from the video model (0 for fake, 1 for real).
        audio_prediction (int): Prediction from the audio model (0 for fake, 1 for real).
    
    Returns:
        int: Final decision (0 for fake, 1 for real).
    """
    return max(video_prediction, audio_prediction)

def main():
    # Example usage of the combine_predictions function
    video_pred = 1  # Assume video model predicts 'real'
    audio_pred = 0  # Assume audio model predicts 'fake'
    
    final_decision = combine_predictions(video_pred, audio_pred)
    print(f"Final decision: {'real' if final_decision == 1 else 'fake'}")

if __name__ == "__main__":
    main()