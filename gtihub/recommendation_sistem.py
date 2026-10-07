"""Sistema sencillo de recomendación de productos con KNeighborsClassifier.

El ejemplo usa un dataset incluido en el propio script para que pueda
ejecutarse sin archivos adicionales. Las características de cada producto son:

    [precio, valoración, compras_recientes, descuento_porcentaje]
"""

from typing import Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Cada etiqueta representa el tipo de producto que se recomendará.
PRODUCT_FEATURES = np.array(
    [
        [25, 4.8, 950, 20],
        [30, 4.6, 820, 15],
        [22, 4.7, 1100, 25],
        [120, 4.7, 420, 10],
        [150, 4.9, 380, 5],
        [95, 4.5, 510, 15],
        [650, 4.8, 150, 8],
        [800, 4.9, 120, 5],
        [500, 4.6, 210, 12],
        [45, 4.2, 700, 30],
        [75, 4.3, 620, 25],
        [60, 4.1, 760, 35],
    ],
    dtype=float,
)

PRODUCT_LABELS = np.array(
    [
        "accesorio",
        "accesorio",
        "accesorio",
        "electronica",
        "electronica",
        "electronica",
        "premium",
        "premium",
        "premium",
        "hogar",
        "hogar",
        "hogar",
    ]
)


def train_recommendation_model(
    features: np.ndarray = PRODUCT_FEATURES,
    labels: np.ndarray = PRODUCT_LABELS,
) -> Tuple[Pipeline, float]:
    """Divide los datos, entrena el clasificador y devuelve modelo y accuracy."""
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.34,
        random_state=42,
        stratify=labels,
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=3)),
        ]
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    return model, accuracy


def predict_recommendation(
    model: Pipeline, characteristics: Sequence[float]
) -> str:
    """Predice el tipo de producto recomendado para sus características."""
    values = np.asarray(characteristics, dtype=float)
    if values.shape != (PRODUCT_FEATURES.shape[1],):
        raise ValueError(
            "Las características deben contener exactamente "
            f"{PRODUCT_FEATURES.shape[1]} valores: "
            "[precio, valoración, compras_recientes, descuento_porcentaje]."
        )

    return str(model.predict(values.reshape(1, -1))[0])


if __name__ == "__main__":
    recommender, accuracy = train_recommendation_model()
    print(f"Accuracy del modelo: {accuracy:.2%}")

    example_product = [40, 4.6, 900, 20]
    recommendation = predict_recommendation(recommender, example_product)
    print(f"Recomendación para {example_product}: {recommendation}")