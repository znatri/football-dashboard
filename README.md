# Football Analytics Dashboard

## Overview
The Football Analytics Dashboard is a comprehensive tool designed to analyze football games using advanced machine learning models. This project leverages YOLO (You Only Look Once) models for object detection and tracking of football players in video footage.

## Features
- **Player Detection**: Detects and tracks football players in video footage.
- **Model Training**: Supports training of YOLOv5 and YOLOv8 models.
- **Environment Setup**: Easy setup of virtual environments and dependencies.
- **Memory Management**: Includes utilities for managing GPU memory.

## Setup

### Prerequisites
- Python 3.10
- `virtualenv` package

### Virtual Environment Setup
To set up the virtual environment and install dependencies, run the following commands:

```bash
virtualenv -p $(which python3.10) --system-site-packages football_env
source football_env/bin/activate
pip install -r requirements.txt
```

### Directory Structure
The project requires the following directories:
- `input_videos`
- `output_videos`
- `stubs`
- `models`
- `memdump`
- `logs`

These directories will be created automatically by the setup script.

## Usage

### Training Models
To train a YOLO model, use the provided Jupyter notebooks in the `training` directory. For example, to train a YOLOv8 model, open and run the `football_training_yolo_v8.ipynb` notebook.

### Memory Management
To deallocate fragmented GPU memory, use the following code:

```bash
python
import torch
torch.cuda.empty_cache()
```

To tweak the environment for better memory management, use:

```bash
python
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author
- **Hardik Goel** - [hardikkgoel@gmail.com](mailto:hardikkgoel@gmail.com)

## Acknowledgments
- [Ultralytics](https://github.com/ultralytics) for the YOLO models.
- [Roboflow](https://roboflow.com) for dataset management.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## Contact
For any questions or issues, please contact [Hardik Goel](mailto:hardikkgoel@gmail.com).