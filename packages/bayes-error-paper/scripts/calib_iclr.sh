for year in {2017..2025}; do
	./scripts/calib.sh config/iclr_${year}.json -o results/iclr_${year}.pdf --ymax 42
done

