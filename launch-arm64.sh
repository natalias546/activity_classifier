#!/bin/bash

## Launch script for ARM64 machines (Apple Silicon, etc.).
## macOS/Linux shortcut: ln -s ~/launch-arm64.sh /usr/local/bin/launch

function launch_usage() {
  echo "Usage: $0 [-t tag (version)] [-d directory]"
  echo "  -t, --tag         Docker image tag (version) to use"
  echo "  -d, --directory   Base directory to use"
  echo "  -h, --help        Print help and exit"
  echo ""
  echo "Example: $0 --tag 2026.1 --directory ~/project_1"
  echo ""
  exit 1
}

while [[ "$#" > 0 ]]; do case $1 in
  -t|--tag) ARG_TAG="$2"; shift;shift;;
  -d|--directory) ARG_DIR="$2";shift;shift;;
  -h|--help) launch_usage;shift; shift;;
  *) echo "Unknown parameter passed: $1"; echo ""; launch_usage; shift; shift;;
esac; done

function finish {
  if [ "$ARG_HOME" != "" ]; then
    echo "Removing empty files and directories ..."
    find "$ARG_HOME" -empty -type d -delete
    find "$ARG_HOME" -empty -type f -delete
  fi
}
trap finish EXIT

ARG_HOME=""
IMAGE_VERSION="latest"
ID="foyie"
LABEL="jupyter-spark-dask-arm64"
IMAGE=${ID}/${LABEL}

if [ "$ARG_TAG" != "" ]; then
  IMAGE_VERSION="$ARG_TAG"
fi

ostype=$(uname)

CPORT=$(curl -s localhost:8787 2>/dev/null)
if [ "$CPORT" != "" ]; then
  echo "-----------------------------------------------------------------------"
  echo "A launch script may already be running. To close the new session and"
  echo "continue with the previous session press q + enter. To continue with"
  echo "the new session and stop the previous session, press enter."
  echo "-----------------------------------------------------------------------"
  read contd
  if [ "${contd}" == "q" ]; then
    exit 1
  fi
fi

clear
has_docker=$(which docker)
if [ "${has_docker}" == "" ]; then
  echo "-----------------------------------------------------------------------"
  echo "Docker is not installed. Download and install Docker from"
  if [[ "$ostype" == "Linux" ]]; then
    echo "https://docs.docker.com/engine/install/"
  elif [[ "$ostype" == "Darwin" ]]; then
    echo "https://docs.docker.com/desktop/install/mac-install/"
  else
    echo "https://docs.docker.com/desktop/install/windows-install/"
  fi
  echo "-----------------------------------------------------------------------"
  read
else
  {
    docker ps -q 2>/dev/null
  } || {
    if [[ "$ostype" == "Darwin" ]]; then
      open /Applications/Docker.app
      while (! docker stats --no-stream 2>/dev/null); do
        echo "Please wait while Docker starts up ..."
        sleep 1
      done
    else
      echo "-----------------------------------------------------------------------"
      echo "Docker is not running. Please start Docker and press [ENTER] to continue."
      echo "-----------------------------------------------------------------------"
      read
    fi
  }

  available=$(docker images -q ${IMAGE}:${IMAGE_VERSION})
  if [ "${available}" == "" ]; then
    echo "-----------------------------------------------------------------------"
    echo "Downloading the ${LABEL}:${IMAGE_VERSION} computing environment ..."
    echo "-----------------------------------------------------------------------"
    docker pull ${IMAGE}:${IMAGE_VERSION}
  fi

  BUILD_DATE=$(docker inspect -f '{{.Created}}' ${IMAGE}:${IMAGE_VERSION})

  echo "-----------------------------------------------------------------------"
  echo "Starting the ${LABEL} computing environment on ${ostype}"
  echo "Build date : ${BUILD_DATE//T*/}"
  echo "Working dir: ${PWD}"
  echo "Ports      : JupyterLab=8888  SparkUI=4040/4041  Dask=8787/8788  HDFS=9870"
  echo "-----------------------------------------------------------------------"

  {
    docker run --rm -it --init \
      -p 127.0.0.1:8888:8888 \
      -p 127.0.0.1:4040:4040 \
      -p 127.0.0.1:4041:4041 \
      -p 127.0.0.1:8787:8787 \
      -p 127.0.0.1:8788:8788 \
      -p 127.0.0.1:9870:9870 \
      -v "${PWD}":/home/work \
      ${IMAGE}:${IMAGE_VERSION}
  } || {
    echo "-----------------------------------------------------------------------"
    echo "It seems there was a problem starting the docker container. Please"
    echo "report the issue and add a screenshot of any messages shown on screen."
    echo "Press [ENTER] to continue."
    echo "-----------------------------------------------------------------------"
    read
  }
fi
