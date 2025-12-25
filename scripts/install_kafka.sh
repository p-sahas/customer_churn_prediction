#!/bin/bash

# Install Kafka for Week 09 Production ML System
# This script installs Apache Kafka natively (without Docker)

set -e

echo " Installing Apache Kafka for ML Pipeline"
echo "==========================================="

# Check if we're on macOS or Linux
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo " Detected macOS - using Homebrew"
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo " Homebrew not found. Please install Homebrew first:"
        echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
    
    echo " Installing Java 17 (required for Kafka)..."
    brew install openjdk@17 || echo "Java might already be installed"
    
    echo " Installing Apache Kafka..."
    brew install kafka
    
    # Get the Kafka installation path
    KAFKA_PATH="$(brew --prefix kafka)/libexec"
    
    echo " Kafka installed at: $KAFKA_PATH"
    echo ""
    echo " Setting up environment variables..."
    
    # Add to shell profile
    SHELL_PROFILE=""
    if [[ "$SHELL" == *"zsh"* ]]; then
        SHELL_PROFILE="$HOME/.zshrc"
    elif [[ "$SHELL" == *"bash"* ]]; then
        SHELL_PROFILE="$HOME/.bashrc"
    fi
    
    if [[ -n "$SHELL_PROFILE" ]]; then
        echo "# Apache Kafka for ML Pipeline" >> "$SHELL_PROFILE"
        echo "export KAFKA_HOME=\"$KAFKA_PATH\"" >> "$SHELL_PROFILE"
        echo "export PATH=\"\$KAFKA_HOME/bin:\$PATH\"" >> "$SHELL_PROFILE"
        echo ""
        echo " Added environment variables to $SHELL_PROFILE"
        echo ""
        echo " To apply changes immediately, run:"
        echo "   export KAFKA_HOME=\"$KAFKA_PATH\""
        echo "   export PATH=\"\$KAFKA_HOME/bin:\$PATH\""
    fi
    
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo " Detected Linux - manual installation"
    
    # Check if Java is installed
    if ! command -v java &> /dev/null; then
        echo " Installing Java 17..."
        sudo apt update
        sudo apt install -y openjdk-17-jdk
    else
        echo " Java is already installed."
    fi
    
    echo " Downloading Apache Kafka..."
    KAFKA_VERSION="3.7.0"
    SCALA_VERSION="2.13"
    FILENAME="kafka_${SCALA_VERSION}-${KAFKA_VERSION}.tgz"
    # UPDATED URL: Use archive.apache.org to prevent 404s on older versions
    DOWNLOAD_URL="https://archive.apache.org/dist/kafka/$KAFKA_VERSION/$FILENAME"
    
    cd ~
    
    # Clean up previous broken downloads if they exist
    if [[ -f "$FILENAME" ]]; then
        echo " Found existing file. verifying..."
        # Check if it is a valid gzip file
        if ! gzip -t "$FILENAME" &>/dev/null; then
            echo " Existing file is corrupt (probably HTML). Deleting and redownloading..."
            rm "$FILENAME"
        fi
    fi

    # Download if not present
    if [[ ! -f "$FILENAME" ]]; then
        echo " Fetching from Archive: $DOWNLOAD_URL"
        # -f fails on HTTP errors, -L follows redirects
        if curl -fLO "$DOWNLOAD_URL"; then
            echo " Download successful."
        else
            echo " Error: Download failed. Check your internet connection."
            exit 1
        fi
    fi
    
    echo " Extracting Kafka..."
    tar -xzf "$FILENAME"
    
    # Move to standard location
    if [[ -d "kafka" ]]; then
        echo " Removing old 'kafka' directory..."
        rm -rf kafka
    fi
    
    # Rename extracted folder to generic 'kafka'
    mv "kafka_${SCALA_VERSION}-${KAFKA_VERSION}" kafka
    
    KAFKA_PATH="$HOME/kafka"
    echo " Kafka installed at: $KAFKA_PATH"
    
    # Add to shell profile
    SHELL_PROFILE="$HOME/.bashrc"
    
    # Avoid duplicate entries in .bashrc
    if ! grep -q "KAFKA_HOME" "$SHELL_PROFILE"; then
        echo "" >> "$SHELL_PROFILE"
        echo "# Apache Kafka for ML Pipeline" >> "$SHELL_PROFILE"
        echo "export KAFKA_HOME=\"$KAFKA_PATH\"" >> "$SHELL_PROFILE"
        echo "export PATH=\"\$KAFKA_HOME/bin:\$PATH\"" >> "$SHELL_PROFILE"
        echo " Added environment variables to $SHELL_PROFILE"
    else
        echo " Environment variables already exist in $SHELL_PROFILE"
    fi
    
    echo ""
    echo " To apply changes immediately, run:"
    echo "   source ~/.bashrc"
    
else
    echo " Unsupported operating system: $OSTYPE"
    echo "Please install Kafka manually from: https://kafka.apache.org/downloads"
    exit 1
fi

echo ""
echo " Kafka installation completed!"
echo ""
echo " Next steps:"
echo "   1. Run: source ~/.bashrc"
echo "   2. Run: make kafka-validate"
echo "   3. Run: make kafka-format"  
echo "   4. Run: make kafka-start"
echo ""
echo " For more details, see: README_KAFKA.md"
