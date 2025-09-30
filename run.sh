#!/usr/bin/env bash
set -euo pipefail

function usage() {
    echo "Usage: $0 --config <config_file> --input <input_path> --output <output_path> [--overwrite] [--without_compress] [--synthetic] [--generate-bbox-from-cuboid]"
    echo ""
    echo "Arguments:"
    echo "  --config                      Path to config yaml file (required)"
    echo "  --input                       Path to input data directory (required)"
    echo "  --output                      Path to output data directory (required)"
    echo "  --overwrite                   Overwrite existing output files (optional)"
    echo "  --without_compress            Do not compress output (optional)"
    echo "  --synthetic                   Convert synthetic data (optional)"
    echo "  --generate-bbox-from-cuboid   Generate 2D bbox from 3D cuboid (optional)"
    exit 1
}

CONFIG_FILE=""
INPUT_PATH=""
OUTPUT_PATH=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_FILE="$(realpath "$2")"
            shift 2
            ;;
        --input)
            INPUT_PATH="$(realpath "$2")"
            shift 2
            ;;
        --output)
            OUTPUT_PATH="$(realpath "$2")"
            shift 2
            ;;
        --overwrite)
            EXTRA_ARGS+=("--overwrite")
            shift
            ;;
        --without_compress)
            EXTRA_ARGS+=("--without_compress")
            shift
            ;;
        --synthetic)
            EXTRA_ARGS+=("--synthetic")
            shift
            ;;
        --generate-bbox-from-cuboid)
            EXTRA_ARGS+=("--generate-bbox-from-cuboid")
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

if [[ -z "$CONFIG_FILE" || -z "$INPUT_PATH" || -z "$OUTPUT_PATH" ]]; then
    echo "Error: --config, --input, and --output are required."
    usage
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Config file '$CONFIG_FILE' does not exist."
    exit 1
fi

if [[ ! -d "$INPUT_PATH" ]]; then
    echo "Error: Input path '$INPUT_PATH' does not exist."
    exit 1
fi

mkdir -p "$OUTPUT_PATH"

echo "Running conversion with:"
echo "  config: $CONFIG_FILE"
echo "  input : $INPUT_PATH"
echo "  output: $OUTPUT_PATH"
echo "  args  : ${EXTRA_ARGS[*]:-none}"
echo ""

# Build the command with proper quoting
CMD="sed 's|input_base:.*|input_base: /data/input|g; s|output_base:.*|output_base: /data/output|g' /config_orig.yaml > /config.yaml && python3 -m perception_dataset.convert --config /config.yaml"
for arg in "${EXTRA_ARGS[@]}"; do
    CMD="$CMD $arg"
done

docker run -it --rm \
  --entrypoint bash \
  -v "${CONFIG_FILE}:/config_orig.yaml:ro" \
  -v "${INPUT_PATH}:/data/input:ro" \
  -v "${OUTPUT_PATH}:/data/output" \
  tier4-perception-converter:latest \
  -c "source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash && $CMD"