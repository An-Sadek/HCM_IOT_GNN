python test/forecast_all.py --checkpoint result/_htgnn1/htgnn_best.pt --output result/_htgnn1/forecast_all.csv --device cuda --batch-size 16
python test/forecast_all.py --checkpoint result/_htgnn2/htgnn_best.pt --output result/_htgnn2/forecast_all.csv --device cuda --batch-size 16
python test/forecast_all.py --checkpoint result/_htgnn3/htgnn_best.pt --output result/_htgnn3/forecast_all.csv --device cuda --batch-size 16
python test/forecast_all.py --checkpoint result/_htgnn4/htgnn_best.pt --output result/_htgnn4/forecast_all.csv --device cuda --batch-size 16

python test/forecast_all.py --checkpoint result/_sehtgnn1/sehtgnn_best.pt --output result/_sehtgnn1/forecast_all.csv --device cuda --batch-size 16
python test/forecast_all.py --checkpoint result/_sehtgnn2/sehtgnn_best.pt --output result/_sehtgnn2/forecast_all.csv --device cuda --batch-size 16
