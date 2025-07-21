# Real-Time Voice Agent with OpenAI

This project is a Python-based voice agent that uses OpenAI's real-time API to have a spoken conversation with an assistant. It uses `aiortc` for WebRTC communication and `pyaudio` for handling microphone input and speaker output.

## Features

- Real-time, low-latency voice conversation.
- Push-to-talk functionality using the spacebar.
- Securely connects to OpenAI's session-based API.

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd VoiceAgentC
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # On Windows
    python -m venv venv
    .\venv\Scripts\activate

    # On macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create a `.env` file:**
    Create a file named `.env` in the root of the project and add your OpenAI API key to it:
    ```
    OPENAI_API_KEY="sk-..."
    ```

5.  **Run the application:**
    ```bash
    python main.py
    ```

## How to Use

- Run the `main.py` script.
- Hold down the **spacebar** to speak to the assistant.
- Release the spacebar to send your audio for processing.
- The assistant's response will be played through your speakers.
- Press `Ctrl+C` in the terminal to exit the application.
