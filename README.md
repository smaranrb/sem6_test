# LLM-based Finance Agent

An intelligent agent utilizing Large Language Models (LLMs) for automated financial news retrieval and stock price prediction.

## Features

- Real-time financial news collection and summarization
- Stock price data analysis and technical indicators
- Sentiment analysis of financial news
- Price prediction using advanced ML models
- Interactive Streamlit dashboard
- LLM-powered insights and recommendations

## Project Structure

```
finance-agent/
├── README.md
├── requirements.txt
├── config.yaml
├── app.py                      # Main Streamlit application
├── data/
│   ├── raw/                    # Raw financial data
│   ├── processed/              # Processed datasets
│   └── models/                 # Saved models
├── src/
│   ├── news/                   # News collection and processing
│   ├── market/                 # Market data analysis
│   ├── models/                 # ML models
│   └── utils/                  # Utility functions
├── tests/                      # Unit tests
└── notebooks/                  # Jupyter notebooks
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/finance-agent.git
cd finance-agent
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure API keys:
- Copy `config.yaml.example` to `config.yaml`
- Add your API keys for news and LLM services

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Access the dashboard at `http://localhost:8501`

## Development

- Run tests: `pytest tests/`
- Format code: `black src/ tests/`
- Lint code: `flake8 src/ tests/`

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request 