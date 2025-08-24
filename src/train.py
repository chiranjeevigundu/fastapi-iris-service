from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path

def main():
    X, y = load_iris(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(Xtr, ytr)
    acc = clf.score(Xte, yte)
    print(f"Test accuracy: {acc:.3f}")
    joblib.dump(clf, Path(__file__).parent / "model.joblib")

if __name__ == "__main__":
    main()
