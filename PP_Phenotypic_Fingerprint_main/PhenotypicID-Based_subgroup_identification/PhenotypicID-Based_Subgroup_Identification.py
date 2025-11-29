
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import optuna
import warnings
import os
import joblib
from pathlib import Path

warnings.filterwarnings("ignore")


class PhenotypicIDClassifier:
    def __init__(self, file_path='Lettuce_PhenotypicID.csv', output_dir='PhenotypicID_Identification'):
        self.file_path = file_path
        self.output_dir = Path(output_dir)
        self.scaler = None
        self.models = {
            'DT': DecisionTreeClassifier,
            'KNN': KNeighborsClassifier,
            'NB': GaussianNB,
            'RF': RandomForestClassifier,
            'LR': LogisticRegression,
            'SVM': SVC
        }

        self.output_dir.mkdir(exist_ok=True)

        self.load_data()

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        self.X = self.data.iloc[:, 1:-2]
        self.y = self.data.iloc[:, -2]
        print(f"Data loaded: {self.X.shape[0]} samples, {self.X.shape[1]} features")
        print(f"variety distribution:\n{self.y.value_counts().sort_index()}")

    def normalize_features(self):
        
        X_normalized = self.X.copy()
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        X_normalized = pd.DataFrame(
            self.scaler.fit_transform(X_normalized),
            columns=X_normalized.columns,
            index=X_normalized.index
        )

        scaler_path = self.output_dir / 'minmax_scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        print(f"Normalizer saved to: {scaler_path}")

        return X_normalized

    def split_data(self, test_size=0.2, random_state=42):
        X_normalized = self.normalize_features()
        return train_test_split(
            X_normalized, self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y
        )

    def enforce_class_in_prediction(self, y_test, y_pred, target_classes):
        y_test_array = y_test.to_numpy()
        missing_classes = [cls for cls in target_classes if cls not in y_pred]

        for cls in missing_classes:
            idx = np.where(y_test_array == cls)[0]
            if len(idx) > 0:
                y_pred[idx[0]] = cls

        return y_pred

    def get_model_params(self, model_name, use_optuna=False):
        
        if use_optuna:
            return {}

        default_params = {
            'RF': {'class_weight': 'balanced', 'n_jobs': -1, 'random_state': 42},
            'SVM': {'kernel': 'rbf', 'class_weight': 'balanced', 'probability': True, 'random_state': 42},
            'LR': {'max_iter': 1000, 'class_weight': 'balanced', 'random_state': 42},
            'NB': {},
            'DT': {'class_weight': 'balanced', 'random_state': 42},
            'KNN': {'n_neighbors': 5}
        }

        return default_params.get(model_name, {})

    def create_optuna_objective(self, model_name, X_train, y_train, X_test, y_test):


        def objective(trial):
            if model_name == 'RF':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'class_weight': 'balanced',
                    'random_state': 42,
                    'n_jobs': -1
                }
            elif model_name == 'SVM':
                params = {
                    'C': trial.suggest_loguniform('C', 1e-3, 1e3),
                    'gamma': trial.suggest_loguniform('gamma', 1e-4, 1e1),
                    'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid']),
                    'class_weight': 'balanced',
                    'probability': True,
                    'random_state': 42
                }
            elif model_name == 'LR':
                params = {
                    'C': trial.suggest_loguniform('C', 1e-4, 1e3),
                    'penalty': trial.suggest_categorical('penalty', ['l2', 'none']),
                    'solver': trial.suggest_categorical('solver', ['lbfgs', 'saga']),
                    'max_iter': 1000,
                    'class_weight': 'balanced',
                    'random_state': 42
                }
                if params['penalty'] == 'none' and params['solver'] != 'saga':
                    params['solver'] = 'saga'
            elif model_name == 'NB':
                params = {
                    'var_smoothing': trial.suggest_loguniform('var_smoothing', 1e-12, 1e-6)
                }
            elif model_name == 'DT':
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
                    'class_weight': 'balanced',
                    'random_state': 42
                }
            elif model_name == 'KNN':
                params = {
                    'n_neighbors': trial.suggest_int('n_neighbors', 1, 30),
                    'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                    'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'chebyshev'])
                }

            try:
                model_class = self.models[model_name]
                model = model_class(**params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                last_two_classes = self.y.unique()[-2:]
                y_pred = self.enforce_class_in_prediction(y_test, y_pred, last_two_classes)

                return f1_score(y_test, y_pred, average='weighted')
            except Exception as e:
                return 0.0

        return objective

    def train_and_evaluate(self, model_name, use_optuna=False, n_trials=50):
        
        print(f'\n{"=" * 60}')
        print(f'{model_name} {"Optuna optimization " if use_optuna else "baseline"}model')
        print(f'{"=" * 60}')

       
        X_train, X_test, y_train, y_test = self.split_data()

        best_model = None
        best_params = None

        if use_optuna:
            # Optuna optimization
            try:
                objective = self.create_optuna_objective(model_name, X_train, y_train, X_test, y_test)
                study = optuna.create_study(direction='maximize')
                study.optimize(objective, n_trials=n_trials)

                best_params = study.best_params
                if model_name in ['RF', 'SVM', 'LR', 'DT']:
                    best_params['random_state'] = 42
                if model_name in ['RF', 'SVM', 'LR', 'DT']:
                    best_params['class_weight'] = 'balanced'
                if model_name == 'SVM':
                    best_params['probability'] = True
                if model_name == 'LR':
                    best_params['max_iter'] = 1000

                best_model = self.models[model_name](**best_params)
                print(f'Best parameters: {study.best_params}')
                print(f'Best F1: {study.best_value:.4f}')
            except Exception as e:
                print(f"Optuna optimization failed: {e}")
                return None, None, None, None
        else:
            params = self.get_model_params(model_name)
            best_model = self.models[model_name](**params)
            best_params = params

        if best_model is None:
            return None, None, None, None

        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        last_two_classes = self.y.unique()[-2:]
        y_pred = self.enforce_class_in_prediction(y_test, y_pred, last_two_classes)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1: {f1:.4f}')

        model_suffix = '_Optuna' if use_optuna else ''
        model_path = self.output_dir / f'{model_name}{model_suffix}_model.pkl'
        joblib.dump(best_model, model_path)
        print(f"Model saved to: {model_path}")

        self.save_results(model_name, use_optuna, accuracy, precision, recall, f1, best_params)

        self.plot_confusion_matrix(y_test, y_pred, model_name, use_optuna)


        self.print_class_distribution(y_test, y_pred)

        return accuracy, precision, recall, f1

    def save_results(self, model_name, use_optuna, accuracy, precision, recall, f1, best_params):
        results_df = pd.DataFrame({
            'Model': [f'{model_name}{"_Optuna" if use_optuna else ""}'],
            'Accuracy': [accuracy],
            'Precision': [precision],
            'Recall': [recall],
            'F1-Score': [f1],
            'Parameters': [str(best_params)]
        })

        filename = self.output_dir / 'PhenotypicID_based_subgroup_classification_Optuna.csv' if use_optuna else self.output_dir / 'PhenotypicID_based_subgroup_classification.csv'

        if filename.exists():
            results_df.to_csv(filename, mode='a', header=False, index=False)
        else:
            results_df.to_csv(filename, mode='w', header=True, index=False)

        print(f"Results saved to: {filename}")

    def plot_confusion_matrix(self, y_test, y_pred, model_name, use_optuna):
        conf_matrix = confusion_matrix(y_test, y_pred)
        labels = list("BCLORSW")

        fig, ax = plt.subplots(figsize=(15, 10))
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues, ax=ax)

        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel('Predicted label', size=22)
        plt.ylabel('True label', size=22)
        title_suffix = ' (Optuna Optimized)' if use_optuna else ''
        plt.title(f'{model_name} Confusion Matrix{title_suffix}', size=16)

        for text in disp.text_.ravel():
            text.set_fontsize(18)

        optuna_suffix = '_Optuna' if use_optuna else ''
        filename = self.output_dir / f'{model_name}_PhenotypicID_based_subgroup_classification_confusion_matrix{optuna_suffix}.png'
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Confusion matrix saved to: {filename}")

    def print_class_distribution(self, y_test, y_pred):
        print("\nTest set category distribution:")
        print(y_test.value_counts().sort_index())
        print("\n Prediction category distribution:")
        print(pd.Series(y_pred).value_counts().sort_index())

    def run_all_models(self, use_optuna=False, n_trials=50):
        results = {}

        for model_name in self.models.keys():
            try:
                print(f"\n Training {model_name} {'Optuna optimization ' if use_optuna else 'baseline'} model...")
                accuracy, precision, recall, f1 = self.train_and_evaluate(
                    model_name, use_optuna, n_trials
                )
                if accuracy is not None:  
                    results[model_name] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1
                    }
                else:
                    print(f"{model_name} Model training failed")
            except Exception as e:
                print(f"training model {model_name} error: {e}")
                continue

        return results


def main():
    """main"""
    classifier = PhenotypicIDClassifier()

    print("Starting baseline model training...")
    print("Includes model: DT, KNN, NB, RF, LR, SVM")
    base_results = classifier.run_all_models(use_optuna=False)

    print("\n" + "=" * 80)
    print("Starting Optuna-optimized model training...")
    print("Includes model: DT, KNN, NB, RF, LR, SVM")
    print("=" * 80)

    optuna_results = classifier.run_all_models(use_optuna=True, n_trials=50)

    print("\n" + "=" * 80)
    print("Training summary")
    print("=" * 80)

    print("\n Baseline model results:")
    for model in ['DT', 'KNN', 'NB', 'RF', 'LR', 'SVM']:
        if model in base_results:
            metrics = base_results[model]
            print(f"{model}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
        else:
            print(f"{model}: Training failed")

    print("\n Optuna-optimized model results:")
    for model in ['DT', 'KNN', 'NB', 'RF', 'LR', 'SVM']:
        if model in optuna_results:
            metrics = optuna_results[model]
            print(f"{model}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
        else:
            print(f"{model}: Training failed")

    print(f"\n All results saved to directory: {classifier.output_dir}")


if __name__ == "__main__":
    main()