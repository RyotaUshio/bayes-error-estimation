for year in {2017..2025}; do
	./scripts/calib.sh config/calib/iclr_${year}.json -o results/calib/iclr_${year}.pdf --ymax 42
done
