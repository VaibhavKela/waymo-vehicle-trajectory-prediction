import pandas as pd

# Store baseline model results
results = {
    "Model": [
        "Linear Regression",
        "Constant Velocity"
    ],
    "ADE (meters)": [
        9.3979,
        12.5612
    ],
    "FDE (meters)": [
        27.0290,
        32.7582
    ]
}

# Create a DataFrame
df = pd.DataFrame(results)

# Calculate percentage improvement over Constant Velocity
ade_improvement = (
    (12.5612 - 9.3979) / 12.5612
) * 100

fde_improvement = (
    (32.7582 - 27.0290) / 32.7582
) * 100

print("BASELINE MODEL COMPARISON")
print("==============================")
print(df.to_string(index=False))

print(f"\nADE improvement: {ade_improvement:.2f}%")
print(f"FDE improvement: {fde_improvement:.2f}%")

# Save results
df.to_csv("outputs/model_comparison.csv", index=False)

print("\nResults saved to outputs/model_comparison.csv")