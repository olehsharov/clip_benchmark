if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3.10 -m venv .venv || exit 1
    . .venv/bin/activate
    echo "Installing dependencies..."
    pip install -r requirements.txt || exit 1
fi
