# SmarTest - AI Project

SmarTest is a local artificial intelligence application developed for the automatic generation and evaluation of exam problems. The project uses deterministic algorithms for problem logic and NLP models (SBERT) for semantic evaluation of answers.

## 📝 Features

- **Problem Generation:**
  - **Games (Nash Equilibrium):** Generates 2x2 game matrices and calculates the Nash equilibrium.
  - **Search (N-Queens):** Generates configurations for the N-Queens problem.
- **Automatic Evaluation:**
  - Exact verification of mathematical solutions.
  - Semantic evaluation of explanations using `sentence-transformers` (SBERT) and cosine similarity.
  - Format validation using regular expressions (Regex).
- **PDF Export:**
  - Ability to download generated problems and results in PDF format.
- **Graphical Interface:**
  - Modern and interactive UI built with Streamlit.

## 🛠️ Project Structure

```
ProiectAI/
├── app/
│   ├── evaluator/      # Modules for answer evaluation (SBERT, Regex)
│   ├── gui/            # Graphical interface components
│   ├── modules/        # Problem logic (Nash, N-Queens, CSP, etc.)
│   └── utils/          # Utilities (PDF Generation, text processing)
├── config.py           # Configuration file
├── main.py             # Application entry point
├── requirements.txt    # Python dependencies
└── README.md           # Documentation
```

## 🚀 Installation and Setup

Follow the steps below to run the application on your local machine.

### 1. Clone the repository
If you have access to the repository, clone it locally:
```bash
git clone <url-repository>
cd ProiectAI
```

### 2. Create a virtual environment (Optional, but recommended)
It is recommended to use a virtual environment to isolate project dependencies.

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
Install the necessary libraries listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Run the application
Start the Streamlit server:
```bash
streamlit run main.py
```

The application will automatically open in your default browser at `http://localhost:8501`.

## 📦 Technologies Used

- **Python 3.x**
- **Streamlit:** For the web interface.
- **Pandas & NumPy:** For data manipulation and numerical calculations.
- **Sentence-Transformers (SBERT):** For semantic text evaluation.
- **Scikit-learn:** For cosine similarity calculation.
- **FPDF:** For generating PDF reports.

## ⚠️ Note
On the first run, the application will download the `all-MiniLM-L6-v2` model for semantic evaluation. This process may take a few moments depending on your internet connection speed.
