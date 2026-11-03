import pandas as pd

results = {
    "Model": [
        "Constant Velocity",
        "Linear Regression - Absolute",
        "Linear Regression - Relative",
        "Linear Regression - Motion Features"
    ],
    "ADE (meters)": [
        12.5612,
        9.3979,
        9.2205,
        10.4624
    ],
    "FDE (meters)": [
        32.7582,
        27.0290,
        26.4291,
        30.4663
    ]
}

df = pd.DataFrame(results)

print("\nCOMPLETE MODEL COMPARISON")
print("=" * 65)
print(df.to_string(index=False))

df.to_csv(
    "outputs/all_model_comparison.csv",
    index=False
)

print("\nResults saved successfully.")