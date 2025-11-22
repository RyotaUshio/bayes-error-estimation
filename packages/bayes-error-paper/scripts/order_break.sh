for x in spearman kendall sigma; do
    uv run -m src.calibration.order_break config/calib/synthetic_logit_gaussian_*[!_binom_noise].json -x ${x} -o results/order_break/synthetic_logit_gaussian_${x}.pdf # --fancy_errorbar 
    uv run -m src.calibration.order_break config/calib/synthetic_logit_gaussian_*_binom_noise.json -x ${x} -o results/order_break/synthetic_logit_gaussian_binom_noise_${x}.pdf # --fancy_errorbar 
done;
