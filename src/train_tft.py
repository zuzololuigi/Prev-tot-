"""Train a Temporal Fusion Transformer model on retail sales data and forecast two weeks ahead."""
from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

LOGGER = logging.getLogger(__name__)


@dataclass
class DataPreparationResult:
    """Container for prepared data artefacts."""

    full_history: pd.DataFrame
    training_dataframe: pd.DataFrame
    prediction_dataframe: pd.DataFrame
    min_date: pd.Timestamp
    max_date: pd.Timestamp


REQUIRED_COLUMNS = {"date", "model", "color", "size", "units_sold"}

QUANTILES = (0.1, 0.5, 0.9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="CSV file with at least 9 months of sales history")
    parser.add_argument("--output", type=Path, default=Path("daily_forecast.csv"), help="Path for the daily forecast CSV")
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("fortnight_summary.csv"),
        help="Path for the aggregated 14-day forecast",
    )
    parser.add_argument("--max-encoder-length", type=int, default=180, help="Maximum number of historical days used by the model")
    parser.add_argument("--max-epochs", type=int, default=60, help="Maximum number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size for training and inference")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of dataloader workers (increase if RAM allows)")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate for the TFT model")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def validate_input(df: pd.DataFrame) -> None:
    missing_cols = REQUIRED_COLUMNS.difference(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")


def _generate_item_id(df: pd.DataFrame) -> pd.Series:
    return (
        df["model"].astype(str).str.strip()
        + "|"
        + df["color"].astype(str).str.strip()
        + "|"
        + df["size"].astype(str).str.strip()
    )


def resolve_data_path(raw_path: Path) -> Path:
    """Resolve the dataset path, being tolerant with names containing spaces."""

    if raw_path.exists():
        return raw_path

    candidate_paths = []

    def _normalise(name: str) -> str:
        return name.lower().replace(" ", "")

    search_roots = []
    if raw_path.parent not in (Path(""), Path(".")):
        search_roots.append(raw_path.parent)
    search_roots.extend([Path.cwd(), Path.cwd() / "data", Path.cwd() / "datasets", Path.cwd() / "Start"])

    desired_names = {raw_path.name}
    if not raw_path.suffix:
        desired_names.add(f"{raw_path.name}.csv")
        desired_names.add(f"{raw_path.name}.CSV")
    desired_normalised = {_normalise(name.split(".")[0]) for name in desired_names}

    for root in search_roots:
        if not root.exists() or not root.is_dir():
            continue
        for file in root.glob("*.csv"):
            candidate_paths.append(file)
            if file.name in desired_names:
                return file
            if _normalise(file.stem) in desired_normalised:
                return file

    if candidate_paths:
        candidates_string = os.linesep.join(str(path) for path in sorted(candidate_paths))
        raise FileNotFoundError(
            f"Unable to locate dataset '{raw_path}'. Available CSV files:\n{candidates_string}"
        )

    raise FileNotFoundError(f"Unable to locate dataset '{raw_path}'. Provide an existing CSV file path.")


def prepare_data(path: Path, prediction_length: int = 14) -> DataPreparationResult:
    df = pd.read_csv(path)
    validate_input(df)

    df["date"] = pd.to_datetime(df["date"], utc=False)
    df.sort_values(["model", "color", "size", "date"], inplace=True)
    df["units_sold"] = df["units_sold"].fillna(0.0).astype(float)

    df["item_id"] = _generate_item_id(df)

    min_date = df["date"].min()
    max_date = df["date"].max()

    LOGGER.info("Data covers %s to %s", min_date.date(), max_date.date())

    # Build a dense daily grid for each SKU to ensure continuity
    full_index = pd.MultiIndex.from_product(
        [df["item_id"].unique(), pd.date_range(start=min_date, end=max_date, freq="D")], names=["item_id", "date"]
    )

    dense_df = df.set_index(["item_id", "date"]).reindex(full_index).sort_index().reset_index()

    # Restore static attributes after reindexing
    static_attributes = df.drop_duplicates("item_id")["item_id model color size".split()].set_index("item_id")
    dense_df = dense_df.join(static_attributes, on="item_id")

    dense_df["units_sold"].fillna(0.0, inplace=True)

    # Add time related features
    dense_df["time_idx"] = (dense_df["date"] - min_date).dt.days
    dense_df["month"] = dense_df["date"].dt.month.astype(int)
    dense_df["day_of_week"] = dense_df["date"].dt.dayofweek.astype(int)
    dense_df["is_weekend"] = (dense_df["day_of_week"] >= 5).astype(int)
    dense_df["week_of_year"] = dense_df["date"].dt.isocalendar().week.astype(int)
    dense_df["year"] = dense_df["date"].dt.year.astype(int)

    # Copy for prediction and extend with future dates
    future_dates = pd.date_range(start=max_date + pd.Timedelta(days=1), periods=prediction_length, freq="D")

    if len(future_dates) == 0:
        raise ValueError("Prediction length must be positive")

    future_records: List[dict] = []
    for item_id, attrs in static_attributes.iterrows():
        for future_date in future_dates:
            future_records.append(
                {
                    "item_id": item_id,
                    "model": attrs["model"],
                    "color": attrs["color"],
                    "size": attrs["size"],
                    "date": future_date,
                    "units_sold": 0.0,
                }
            )

    future_df = pd.DataFrame(future_records)
    future_df["time_idx"] = (future_df["date"] - min_date).dt.days
    future_df["month"] = future_df["date"].dt.month.astype(int)
    future_df["day_of_week"] = future_df["date"].dt.dayofweek.astype(int)
    future_df["is_weekend"] = (future_df["day_of_week"] >= 5).astype(int)
    future_df["week_of_year"] = future_df["date"].dt.isocalendar().week.astype(int)
    future_df["year"] = future_df["date"].dt.year.astype(int)

    combined_df = pd.concat([dense_df, future_df], ignore_index=True, sort=False)

    return DataPreparationResult(
        full_history=dense_df,
        training_dataframe=dense_df,
        prediction_dataframe=combined_df,
        min_date=min_date,
        max_date=max_date,
    )


def create_datasets(
    prepared: DataPreparationResult,
    max_encoder_length: int,
    prediction_length: int = 14,
) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet, TimeSeriesDataSet, int]:
    data = prepared.training_dataframe

    # ensure encoder length is valid
    history_length = int(data["time_idx"].max()) + 1
    encoder_length = min(max_encoder_length, max(prediction_length, history_length - prediction_length))
    encoder_length = max(encoder_length, prediction_length)

    training_cutoff = data["time_idx"].max() - prediction_length

    LOGGER.info("Using encoder length of %s days", encoder_length)

    training_dataset = TimeSeriesDataSet(
        data[data["time_idx"] <= training_cutoff],
        time_idx="time_idx",
        target="units_sold",
        group_ids=["item_id"],
        min_encoder_length=encoder_length // 2,
        max_encoder_length=encoder_length,
        min_prediction_length=prediction_length,
        max_prediction_length=prediction_length,
        static_categoricals=["model", "color", "size"],
        time_varying_known_reals=["time_idx"],
        time_varying_known_categoricals=["month", "day_of_week", "is_weekend", "week_of_year", "year"],
        time_varying_unknown_reals=["units_sold"],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        target_normalizer=GroupNormalizer(groups=["item_id"], transformation="softplus"),
    )

    validation_dataset = TimeSeriesDataSet.from_dataset(training_dataset, data, predict=True, stop_randomization=True)

    prediction_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        prepared.prediction_dataframe,
        predict=True,
        stop_randomization=True,
    )

    return training_dataset, validation_dataset, prediction_dataset, training_cutoff


def train_model(
    training_dataset: TimeSeriesDataSet,
    validation_dataset: TimeSeriesDataSet,
    args: argparse.Namespace,
) -> TemporalFusionTransformer:
    pl.seed_everything(42, workers=True)

    train_loader = training_dataset.to_dataloader(train=True, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = validation_dataset.to_dataloader(train=False, batch_size=args.batch_size, num_workers=args.num_workers)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="tft-best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        auto_insert_metric_name=False,
    )
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=8, min_delta=1e-4, mode="min")

    logger = CSVLogger(save_dir="logs", name="tft")

    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=args.learning_rate,
        hidden_size=64,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=32,
        lstm_layers=2,
        loss=QuantileLoss(),
        output_size=7,
        reduce_on_plateau_patience=4,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        enable_model_summary=True,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=logger,
        log_every_n_steps=10,
    )

    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_model_path = checkpoint_callback.best_model_path
    if not best_model_path:
        raise RuntimeError("Training finished without producing a checkpoint")

    LOGGER.info("Best model checkpoint saved to %s", best_model_path)

    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    return best_tft


def make_predictions(
    model: TemporalFusionTransformer,
    prediction_dataset: TimeSeriesDataSet,
    training_cutoff: int,
    min_date: pd.Timestamp,
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
    predict_loader = prediction_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=num_workers)

    forecasts, index = model.predict(
        predict_loader,
        return_index=True,
        mode="quantiles",
        quantiles=list(QUANTILES),
    )

    predictions = index.copy()
    forecast_array = forecasts.detach().cpu().numpy()
    for column, quantile in zip(forecast_array.T, QUANTILES):
        predictions[f"predicted_units_p{int(quantile * 100):02d}"] = column.reshape(-1)

    predictions = predictions[predictions["time_idx"] > training_cutoff]
    predictions["date"] = predictions["time_idx"].apply(lambda idx: min_date + pd.Timedelta(days=int(idx)))

    return predictions


def enrich_predictions(predictions: pd.DataFrame, prepared: DataPreparationResult) -> pd.DataFrame:
    static_info = prepared.full_history.drop_duplicates("item_id")["item_id model color size".split()]
    enriched = predictions.merge(static_info, on="item_id", how="left")
    prediction_columns = [col for col in enriched.columns if col.startswith("predicted_units_")]
    enriched = enriched[["date", "model", "color", "size", *sorted(prediction_columns), "time_idx"]]
    enriched.sort_values(["date", "model", "color", "size"], inplace=True)
    return enriched


def save_outputs(
    enriched_predictions: pd.DataFrame,
    output_path: Path,
    summary_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    enriched_predictions.to_csv(output_path, index=False)

    quantile_columns = [col for col in enriched_predictions.columns if col.startswith("predicted_units_")]
    summary = (
        enriched_predictions.groupby(["model", "color", "size"], as_index=False)[quantile_columns]
        .sum()
        .rename(columns={
            column: column.replace("predicted_units_", "predicted_units_next_14_days_")
            for column in quantile_columns
        })
    )
    summary.to_csv(summary_path, index=False)

    LOGGER.info("Saved daily forecast to %s and 14-day summary to %s", output_path, summary_path)


def main() -> None:
    args = parse_args()
    configure_logging()

    prediction_length = 14

    data_path = resolve_data_path(args.data)
    LOGGER.info("Loading data from %s", data_path)
    prepared = prepare_data(data_path, prediction_length=prediction_length)

    training_dataset, validation_dataset, prediction_dataset, training_cutoff = create_datasets(
        prepared=prepared,
        max_encoder_length=args.max_encoder_length,
        prediction_length=prediction_length,
    )

    LOGGER.info("Starting model training")
    model = train_model(
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        args=args,
    )

    LOGGER.info("Generating predictions")
    predictions = make_predictions(
        model=model,
        prediction_dataset=prediction_dataset,
        training_cutoff=training_cutoff,
        min_date=prepared.min_date,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    enriched = enrich_predictions(predictions, prepared)
    save_outputs(enriched, args.output, args.summary_output)

    LOGGER.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
