. .venv/bin/activate

if [ ! -d "images" ]; then
    echo "Images directory does not exist. Generating images..."
    python main.py generate --count 10000 || exit 1
fi

echo "Testing..."
python main.py test || exit 1

echo "Done!"
