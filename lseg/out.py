import pandas as pd

def save_predictions_csv(predictions: pd.Series, dates: list[str], output_path: str):
    """
    Saves predictions in the format required by the competition.

    Args:
        predictions: A Series of predicted values (floats), indexed by day or aligned with dates.
        dates: A Series or list of dates (datetime or strings).
        output_path: Path to save the CSV file.
    """
    df_out = pd.DataFrame({
        "date": dates,
        "value": predictions.values,
        "team": ["Andrei Iușan", "Lucian Ursu"] + [""] * (len(predictions) - 2)
    })

    df_out.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to: {output_path}")

