for config_file in config/feebee/*.json; do
    config_name=$(basename "$config_file" .json)
    ./scripts/feebee.sh "$config_file" -o results/feebee/${config_name}.pdf "$@"
done
