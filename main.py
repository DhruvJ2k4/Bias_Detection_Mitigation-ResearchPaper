import pandas as pd
from utils import load_config, set_seed, get_logger
from src.preprocessing import build_preprocessor
from src.models import get_models
from src.evaluation import evaluate_model

def main():
    logger = get_logger("BiasPipeline")
    config = load_config()
    set_seed(config['seed'])

    logger.info("Loading dataset...")
    df = pd.read_csv("data/raw/indian_loan_dataset.csv")

    logger.info("Preprocessing...")
    preprocessor, X_train, X_test, y_train, y_test = build_preprocessor(df)

    logger.info("Training models...")
    models = get_models(config)
    results = {}

    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None
        results[name] = evaluate_model(y_test, y_pred, y_prob)

    pd.DataFrame(results).T.to_csv("results/metrics.csv")
    logger.info("Results saved to results/metrics.csv")

if __name__ == "__main__":
    main()
