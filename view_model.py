
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load our trained model.
saved_data = joblib.load("flight_phase_model.joblib")
model = saved_data["model"]
features = saved_data["features"]

# Display the learned decision rules.
plt.figure(figsize=(22, 12))

plot_tree(
    model,
    feature_names=[
        "altitude_m",
        "vertical_speed_mps",
        "accel_x",
        "accel_y",
        "accel_z",
        "pitch_deg",
        "roll_deg",
    ],
    class_names=model.classes_,
    filled=True,
    rounded=True,
    fontsize=8,
)

plt.title("AeroSense - Learned Decision Tree")
plt.tight_layout()
plt.savefig("decision_tree.png", dpi=150)
plt.show()

print("Tree depth:", model.get_depth())
print("Number of leaves:", model.get_n_leaves())
print("Classes:", model.classes_)
print("Saved decision_tree.png")
