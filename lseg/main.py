from lseg.preprocessing import load_price_data
from lseg.evals import evaluate_ensemble, evaluate_gbm, evaluate_arima, evaluate_baseline, evaluate_fft


def main():
    df = load_price_data('./data/Price_History_2025.xlsx')

    evaluate_ensemble(df)
    evaluate_gbm(df)
    evaluate_arima(df)
    evaluate_baseline(df)
    evaluate_fft(df)

if __name__ == '__main__':
    main()
