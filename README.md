# Swin-Large XAI Research Project

This project implements the Swin-Large architecture for Explainable AI (XAI) research, focusing on the analysis of video and audio content. The goal is to develop models that can effectively determine the authenticity of media, providing insights into the decision-making process.

## Project Structure

```
swin-large-xai-research
├── src
│   ├── model_factory.py       # Core architecture of the Swin-Large model
│   ├── train_video.py         # Training script for video model
│   ├── test_video.py          # Testing script for video model
│   ├── train_audio.py         # Training script for audio model
│   ├── test_audio.py          # Testing script for audio model
│   └── unimodel_or_fusion.py  # Decision-level fusion implementation
├── data
│   ├── video                  # Directory for video data
│   └── audio                  # Directory for audio data
├── models
│   └── checkpoints            # Directory for model checkpoints
├── results
│   ├── logs                   # Directory for training logs
│   └── outputs                # Directory for model outputs
├── requirements.txt           # Required Python packages and dependencies
├── README.md                  # Project documentation
└── .gitignore                 # Files and directories to ignore in version control
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd swin-large-xai-research
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

- To train the video model, run:
  ```
  python src/train_video.py
  ```

- To test the video model, run:
  ```
  python src/test_video.py
  ```

- To train the audio model, run:
  ```
  python src/train_audio.py
  ```

- To test the audio model, run:
  ```
  python src/test_audio.py
  ```

- To perform decision-level fusion, run:
  ```
  python src/unimodel_or_fusion.py
  ```

## Model Descriptions

- **Swin-Large Model**: The core architecture that utilizes the Swin Transformer block with attention weights, designed for XAI research.

- **Video Model**: Trained on face-crop images to classify video content as "fake" or "real."

- **Audio Model**: Trained on Mel-spectrogram images to classify audio content as "fake" or "real."

- **Decision-Level Fusion**: Combines predictions from both video and audio models to enhance the accuracy of content authenticity assessments.

## Contributions

Contributions to this project are welcome. Please submit a pull request or open an issue for discussion.

## License

This project is licensed under the MIT License. See the LICENSE file for details.