for config_file in config/calib/synthetic_logit_gaussian_*.json; do
    config_name=$(basename "$config_file" .json)
    ./scripts/calib.sh "$config_file" -o results/calib/${config_name}.pdf "$@"
done
