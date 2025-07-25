if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3.10 -m venv .venv || exit 1
    . .venv/bin/activate
    echo "Installing dependencies..."
    pip install -r requirements.txt || exit 1
fi
. .venv/bin/activate

if [ ! -d "images" ]; then
    echo "Images directory does not exist. Generating images..."
    python main.py generate --count 10000 || exit 1
fi

echo "Testing..."
python main.py test || exit 1

echo "Done!"
