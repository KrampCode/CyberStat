# CyberStat - Data Analytics Chat Agent

A conversational statistical analyst powered by PydanticAI and Gemini.

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # or: source venv/bin/activate  # macOS/Linux
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure your API key:
   ```bash
   copy .env.example .env
   # Edit .env and set GOOGLE_API_KEY
   # Optionally set CYBERSTAT_MODEL to any PydanticAI model string
   ```

4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Getting a Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key to your `.env` file

## Running Tests

The test suite checks the statistical tools against scipy/sklearn reference values and
exercises the agent's tool wiring with scripted models (no API key needed):

```bash
pip install -r requirements-dev.txt
pytest
```
