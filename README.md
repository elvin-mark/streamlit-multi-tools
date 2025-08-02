# Streamlit Multi-Tools

This repository is a collection of useful tools built with Streamlit. Each tool is a standalone application designed for a specific purpose, from development and finance to entertainment.

## Tools

### Development

- **API Client:** A simple, yet powerful API client similar to Insomnia or Bruno. It allows you to make HTTP requests (GET, POST, PUT, DELETE, PATCH) with custom headers, authentication (Basic, Bearer, API Key), and body (JSON, Raw, Form-data, File Upload). It also provides a history of your requests.
- **DB Explorer:** A database explorer for PostgreSQL. It allows you to connect to a database, view tables and their schemas, and execute SQL queries.

### Entertainment

- **Local Music Player:** A simple music player that scans your local `~/Music` directory and allows you to play your favorite tunes.
- **Local Video Player:** A video player that scans your local `~/Videos` directory and allows you to watch your videos.

### Finance

- **Stock Analysis:** A stock analysis tool that uses Yahoo Finance data to provide you with insights into your favorite stocks. It includes charts for price, moving averages, RSI, and MACD, as well as a simple buy/sell recommendation.

## Getting Started

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/elvin-mark/streamlit-multi-tools.git
    cd streamlit-multi-tools
    ```
2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run a tool:**
    To run a specific tool, use the `streamlit run` command followed by the path to the tool's Python file. For example, to run the API Client:
    ```bash
    streamlit run tools/development/api_client.py
    ```

## Contributing

Contributions are welcome! If you have an idea for a new tool or want to improve an existing one, please open an issue or submit a pull request.
