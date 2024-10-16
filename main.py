import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import seaborn as sns
from sklearn.metrics import confusion_matrix


# Verileri dengelemek için SMOTE uygulaması
def balance_data(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

# Makine öğrenmesi modeli eğitme ve değerlendirme (parametre optimizasyonu ile)
def train_optimized_model(X, y):
    # Veriyi eğitim ve test setlerine ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Verileri dengeleme (SMOTE)
    X_train_balanced, y_train_balanced = balance_data(X_train, y_train)

    # Random Forest parametre grid
    param_grid = {
        'n_estimators': [100, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True],
        'class_weight': ['balanced']
    }

    # Random Forest modelini oluştur
    rf = RandomForestClassifier(random_state=42)

    # Grid Search ile parametre optimizasyonu
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')

    # Modeli eğit
    grid_search.fit(X_train_balanced, y_train_balanced)

    # En iyi parametrelerle eğitimli modeli al
    best_model = grid_search.best_estimator_

    # Test verisi üzerinde tahmin yap
    y_pred = best_model.predict(X_test)

    # Model performansını değerlendirme
    print("Best Parameters:", grid_search.best_params_)
    print("Optimized Accuracy:", accuracy_score(y_test, y_pred))
    print("\nOptimized Classification Report:\n", classification_report(y_test, y_pred))

    # Sadece modeli döndür
    return best_model,X_test,y_test

# Tahmin ve uyarı mesajları oluşturma
def predict_with_custom_threshold(model, new_data, threshold=0.5):
    probabilities = model.predict_proba(new_data)
    predictions = (probabilities[:, 1] >= threshold).astype(int)

    # Uyarı mesajları
    warnings = []
    for i, prob in enumerate(probabilities[:, 1]):
        if prob >= 0.8:
            warnings.append(f"\033[91mMachine {i}: **Acil bakım gerekli** (Arıza olasılığı: {prob:.2f})")
        elif prob >= 0.50:
            warnings.append(f"\033[93mMachine {i}: **Arıza riski yüksek** (Arıza olasılığı: {prob:.2f})")
        else:
            warnings.append(f"\033[92mMachine {i}: Normal (Arıza olasılığı: {prob:.2f})")

    return predictions, warnings

# Özellik önemlerini görselleştirme fonksiyonu
def plot_feature_importances(model, X):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
    plt.tight_layout()
    plt.show()

# Karar ağaçlarını görselleştirme fonksiyonu
def visualize_tree(model, X, tree_index=0):
    plt.figure(figsize=(20, 10))  # Şeklin boyutlarını ayarla
    tree = model.estimators_[tree_index]
    plot_tree(tree, feature_names=X.columns, class_names=["No Fault", "Fault"], filled=True, rounded=True, fontsize=10, max_depth=3)
    plt.title(f"Decision Tree {tree_index + 1}")
    plt.show()


# Confusion Matrix'i oluştur ve görselleştir
def plot_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Fault', 'Fault'],
                yticklabels=['No Fault', 'Fault'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


# Ana fonksiyon
def main(csv_file_path):
    # Verileri yükle
    df = pd.read_csv(csv_file_path)

    # Verileri hazırla: Baskı sayısı, sıcaklık, toner seviyesi, kağıt sıkışma miktarı, haftalık kullanım
    X = df[['Print_Count', 'Temperature', 'Toner_Level', 'Paper_Jam_Count', 'Weekly_Usage_Hours']]
    y = df['Fault_Occurred']

    # Verilerin tamamı üzerinde model eğitimi ve tahmin
    model, X_test, y_test= train_optimized_model(X, y)  # Artık sadece model döndürülüyor

    # Özellik önemini görselleştir
    plot_feature_importances(model, X)

    # Tahmin yap
    predictions, warnings = predict_with_custom_threshold(model, X)

    # Sonuçları orijinal veriyle birleştirerek gösterme
    df['Predicted_Fault'] = predictions

    # Uyarı mesajlarını döndür
    for warning in warnings:
        print(warning)

    # Karar ağacını görselleştir
    visualize_tree(model, X)


    plot_confusion_matrix(model, X_test, y_test)


    return df

# Örnek kullanım (CSV dosyasının yolunu gir)
csv_file_path = 'predictive12.csv'
df_result = main(csv_file_path)

# Sonuçları göstermek için
print(df_result.head())


# Grafik çizimi
plt.figure(figsize=(10, 6))
sns.boxplot(x='Fault_Occurred', y='Temperature', data=df_result)
plt.title('Temperature vs Fault Occurrence')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Fault_Occurred', y='Toner_Level', data=df_result)
plt.title('Toner_Level vs Fault Occurrence')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Fault_Occurred', y='Print_Count', data=df_result)
plt.title('Print_Count vs Fault Occurrence')
plt.show()

# plt.figure(figsize=(10, 6))
# sns.boxplot(x='Fault_Occurred', y='Weekly_Usage', data=df_result)
# plt.title('Weekly_Usage vs Fault Occurrence')
# plt.show()


# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='Print_Count', y='Temperature', hue='Fault_Occurred', data=df_result)
# plt.title('Print Count vs Temperature and Fault Occurrence')
# plt.show()


corr_matrix = df_result.corr()
plt.figure(figsize=(12, 10))  # Grafik boyutunu artır
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')  # '.2f' ile noktadan sonra 2 basamak gösterilir
plt.title('Correlation Matrix')
plt.tight_layout()  # Sıkışık yerleşimi düzelt
plt.show()
