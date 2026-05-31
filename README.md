# activity_classifier
UCI Repository movement from wearable device data classification 

## Getting Started with Docker

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

### Launch the Container

From your DSE 230 directory, run the launch script:

```bash
bash "launch-arm64.sh" -d /path/to/your/project
```

The script will automatically start Docker Desktop if it isn't running, pull the image if needed, and mount your current directory into the container.

**Optional flags:**
- `-t` / `--tag` — specify an image version (default: `latest`), e.g. `--tag 2026.1`
- `-d` / `--directory` — base directory to mount

**Example:**
```bash
bash "launch-arm64.sh" --tag 2026.1 --directory ~/activity_classifier
```

### Edit Code in VS Code
Your local directory is mounted into the container at `/home/work`, so any edits you make in VS Code are reflected inside the container in real time — no extra setup needed.

### Stop the Container
Press `Ctrl + C` in the terminal where the script is running.



## Code flow for the CNN-LSTM model

### Run eda.ipynb
This will do some initial data preperation (like loading the files) and also some eda

### Run datawrangling.ipynb
This will setup and create all the datasets that will be used later on. Make sure to hold on to the .parquet files that were produced.

### Run cnn_lstm_classifier.ipynb
Take the .parquet files that were produced and put them in the same directory as cnn_lstm_classifier, or update the "base" file path in cnn_lstm_classifier.ipynb, then run it. I suggest running this in gooogle co-lab with the TPU, otherwise it will take a few hours.
