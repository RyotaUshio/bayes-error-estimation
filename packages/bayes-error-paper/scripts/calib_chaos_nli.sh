run() {
    ./scripts/calib.sh config/calib/$1.json -o results/calib/$1.pdf --hline $2
}

run snli 9.709379128137384
run mnli 14.38398999374609
run abduptive_nli 14.68668407310705
