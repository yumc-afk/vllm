

set -e

usage() {
    echo "Usage: $0 [--cpu|--gpu]"
    echo "  --cpu: Install dependencies for CPU-only testing"
    echo "  --gpu: Install dependencies for GPU testing"
    exit 1
}

if [ $# -eq 0 ]; then
    usage
fi

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --cpu)
            CPU_MODE=true
            shift
            ;;
        --gpu)
            GPU_MODE=true
            shift
            ;;
        *)
            usage
            ;;
    esac
done

echo "Creating Python virtual environment..."
python -m venv venv
source venv/bin/activate

echo "Installing common dependencies..."
pip install --upgrade pip
pip install pytest mock

if [ "$CPU_MODE" = true ]; then
    echo "Installing CPU-only dependencies..."
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    
    pip install numpy
    pip install -e . --no-deps
elif [ "$GPU_MODE" = true ]; then
    echo "Installing GPU dependencies..."
    pip install torch --index-url https://download.pytorch.org/whl/cu118
    
    pip install -e .
fi

echo "Installation complete!"
echo "Activate the virtual environment with: source venv/bin/activate"
