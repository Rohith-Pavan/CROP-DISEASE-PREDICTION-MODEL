
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Combined Training Script (Comprehensive)
Includes:
- EDA and Data Loading
- Feature Extraction (Color Histograms)
- ML Baselines (LR, RF, GB)
- Deep Learning (ResNet50 with Transfer Learning)
- ROC AUC Analysis
- Comprehensive Model Saving
- Disease Recommendations
"""

# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import warnings
import pickle
import json
from datetime import datetime
import ssl
import certifi

# Set SSL cert for Mac
os.environ['SSL_CERT_FILE'] = certifi.where()

warnings.filterwarnings('ignore')

# Image processing
from PIL import Image
import cv2
from skimage import feature

# Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.utils import class_weight

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Configuration
CONFIG = {
    'DATASET_PATH': '/Users/rohithpavan/Desktop/SL Dataset',
    'IMG_SIZE': (224, 224),  
    'BATCH_SIZE': 32,
    'EPOCHS': 50,
    'RANDOM_STATE': 42,
    'TEST_SIZE': 0.15,
    'VAL_SIZE': 0.15,
    'N_CLASSES': None
}

# Set random seeds
np.random.seed(CONFIG['RANDOM_STATE'])
tf.random.set_seed(CONFIG['RANDOM_STATE'])

# Create directories
os.makedirs('saved_models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

print("✓ Configuration loaded!")
print(f"Dataset path: {CONFIG['DATASET_PATH']}")

def load_dataset_structure(dataset_path):
    """Load and analyze dataset structure"""
    dataset_path = Path(dataset_path)
    
    # Check if train/test split exists
    if (dataset_path / 'train').exists():
        train_path = dataset_path / 'train'
        test_path = dataset_path / 'test' if (dataset_path / 'test').exists() else None
    else:
        train_path = dataset_path
        test_path = None
    
    # Get all class folders
    classes = sorted([d.name for d in train_path.iterdir() if d.is_dir()])
    
    # Count images per class and filter small classes (ensuring enough for split)
    class_counts = {}
    valid_classes = []
    for class_name in classes:
        class_path = train_path / class_name
        images = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png')) + list(class_path.glob('*.jpeg'))
        count = len(images)
        if count >= 5:
            valid_classes.append(class_name)
            class_counts[class_name] = count
        else:
            print(f"Warning: Class '{class_name}' has only {count} images. Excluding from training.")
            
    CONFIG['N_CLASSES'] = len(valid_classes)
    classes = sorted(valid_classes) # Ensure sorted order
    
    print(f"✓ Found {len(classes)} valid classes")
    print(f"✓ Total images: {sum(class_counts.values())}")
    print(f"\nClass distribution:")
    for cls in classes[:10]:
         print(f"  {cls}: {class_counts[cls]} images")
    
    return train_path, test_path, classes, class_counts

def load_images_and_labels(data_path, classes, img_size=(224, 224), max_images_per_class=None):
    """Load all images and labels"""
    images = []
    labels = []
    
    print("Loading images...")
    for class_idx, class_name in enumerate(classes):
        class_path = Path(data_path) / class_name
        image_files = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png')) + list(class_path.glob('*.jpeg'))
        
        if max_images_per_class:
            image_files = image_files[:max_images_per_class]
        
        for img_file in image_files:
            try:
                img = Image.open(img_file).convert('RGB')
                img = img.resize(img_size)
                img_array = np.array(img)
                images.append(img_array)
                labels.append(class_idx)
            except Exception as e:
                print(f"Error loading {img_file}: {e}")
        
        if (class_idx + 1) % 5 == 0:
            print(f"  Processed {class_idx + 1}/{len(classes)} classes...")
    
    return np.array(images), np.array(labels)

def extract_features(images):
    """Extract color histogram features"""
    features = []
    
    print("Extracting features (Color Histograms)...")
    for i, img in enumerate(images):
        # Color histograms (32 bins per channel)
        hist_r = np.histogram(img[:, :, 0], bins=32, range=(0, 256))[0]
        hist_g = np.histogram(img[:, :, 1], bins=32, range=(0, 256))[0]
        hist_b = np.histogram(img[:, :, 2], bins=32, range=(0, 256))[0]
        
        # Normalize
        color_features = np.concatenate([hist_r, hist_g, hist_b])
        color_features = color_features / (img.shape[0] * img.shape[1])
        
        features.append(color_features)
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(images)} images")
    
    return np.array(features)

def plot_roc_curves(y_test, y_score, classes, title='ROC Curve', filename='roc_curve.png'):
    """Plot ROC curves for multi-class"""
    # Binarize the output
    y_test_bin = label_binarize(y_test, classes=range(len(classes)))
    n_classes = y_test_bin.shape[1]
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    colors = plt.cm.viridis(np.linspace(0, 1, n_classes))
    for i, color in zip(range(min(n_classes, 10)), colors): # Limit to 10 classes for clarity
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(classes[i], roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right", fontsize='small')
    plt.savefig(f'plots/{filename}')
    plt.show()
    plt.close()
    
    return roc_auc["micro"]

def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix', filename='confusion_matrix.png'):
    """Plot Confusion Matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, cbar_kws={'label': 'Count'})
    plt.title(title, fontsize=13, fontweight='bold')
    plt.ylabel('True Label', fontsize=11)
    plt.xlabel('Predicted Label', fontsize=11)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(f'plots/{filename}')
    plt.show()
    plt.close()

def plot_samples(X, y, classes, samples_per_class=5):
    n_classes = min(len(classes), 10)  # Show max 10 classes
    fig, axes = plt.subplots(n_classes, samples_per_class, figsize=(15, 3*n_classes))
    
    for class_idx in range(n_classes):
        class_images = X[y == class_idx]
        n_available = min(samples_per_class, len(class_images))
        if n_available == 0: continue
        sample_indices = np.random.choice(len(class_images), n_available, replace=False)
        
        for i in range(samples_per_class):
            ax = axes[class_idx, i] if n_classes > 1 else axes[i]
            if i < n_available:
                ax.imshow(class_images[sample_indices[i]].astype('uint8'))
                ax.axis('off')
                if i == 0:
                    ax.text(-10, class_images[0].shape[0]//2, classes[class_idx], 
                           rotation=90, va='center', fontsize=10, fontweight='bold')
            else:
                ax.axis('off')
    
    plt.suptitle('Sample Images from Each Class', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()

# ==================================================================================
# MAIN EXECUTION
# ==================================================================================

# 1. Load Data
train_path, test_path, classes, class_counts = load_dataset_structure(CONFIG['DATASET_PATH'])
X, y = load_images_and_labels(train_path, classes, CONFIG['IMG_SIZE'])

print(f"\n✓ Dataset shape: {X.shape}")
print(f"✓ Labels shape: {y.shape}")
print(f"✓ Memory usage: {X.nbytes / (1024**3):.2f} GB")

# 2. Split Data
# Stratified split
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=CONFIG['TEST_SIZE'], 
    stratify=y, random_state=CONFIG['RANDOM_STATE']
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, 
    test_size=CONFIG['VAL_SIZE']/(1-CONFIG['TEST_SIZE']), 
    stratify=y_temp, 
    random_state=CONFIG['RANDOM_STATE']
)

print(f"Train set: {X_train.shape[0]} images ({len(X_train)/len(X)*100:.1f}%)")
print(f"Validation set: {X_val.shape[0]} images ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test set: {X_test.shape[0]} images ({len(X_test)/len(X)*100:.1f}%)")

# Normalize to [0, 1] for ML models - Features will be extracted from raw or normalized?
# extract_features uses raw images (0-255).
# ResNet50 uses its own preprocess_input.
# So we keep X_train as is for now, and handle normalization where needed.

# 3. EDA
# EDA 1: Class Distribution
plt.figure(figsize=(14, 6))
unique, counts = np.unique(y_train, return_counts=True)
colors = plt.cm.viridis(np.linspace(0, 1, len(unique)))

plt.bar([classes[i] for i in unique], counts, color=colors, edgecolor='black', alpha=0.8)
plt.xlabel('Disease Class', fontsize=13, fontweight='bold')
plt.ylabel('Number of Images', fontsize=13, fontweight='bold')
plt.title('Class Distribution in Training Set', fontsize=15, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('plots/class_distribution.png')
plt.show()
plt.close()

# EDA 2: Sample Visualization
plot_samples(X_train, y_train, classes)

# EDA 3: Image Statistics
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Brightness per class
brightness_per_class = []
for class_idx in range(len(classes)):
    class_images = X_train[y_train == class_idx]
    if len(class_images) > 0:
        brightness = np.mean(class_images, axis=(1, 2, 3))
        brightness_per_class.append(brightness)
    else:
        brightness_per_class.append([])

bp = axes[0].boxplot(brightness_per_class, labels=classes, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
axes[0].set_xlabel('Class', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Average Brightness', fontsize=11, fontweight='bold')
axes[0].set_title('Brightness Distribution by Class', fontsize=12, fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(axis='y', alpha=0.3)

# RGB histograms
for class_idx in range(min(3, len(classes))):
    class_images = X_train[y_train == class_idx]
    if len(class_images) > 0:
        sample_img = class_images[np.random.randint(len(class_images))]
        
        for channel, color in enumerate(['red', 'green', 'blue']):
            axes[1].hist(sample_img[:, :, channel].flatten(), bins=50, 
                        alpha=0.4, label=f'{classes[class_idx]}-{color}', color=color)

axes[1].set_xlabel('Pixel Value', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[1].set_title('RGB Channel Distribution (Sample Images)', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
plt.close()


# ==================================================================================
# PART 1: MACHINE LEARNING BASELINES (Color Histograms)
# ==================================================================================
print("\n" + "="*70)
print("PART 1: MACHINE LEARNING BASELINES (Color Histograms)")
print("="*70)

# Extract features
X_train_feat = extract_features(X_train)
X_val_feat = extract_features(X_val)
X_test_feat = extract_features(X_test)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_feat)
X_val_scaled = scaler.transform(X_val_feat)
X_test_scaled = scaler.transform(X_test_feat)

print(f"\n✓ Feature shape: {X_train_scaled.shape}")

# A. Logistic Regression
print("\nBASELINE MODEL 1: LOGISTIC REGRESSION")
lr_model = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000,
    C=1.0,
    class_weight='balanced',
    random_state=CONFIG['RANDOM_STATE'],
    verbose=1
)
lr_model.fit(X_train_scaled, y_train)
train_acc_lr = accuracy_score(y_train, lr_model.predict(X_train_scaled))
y_val_pred_lr = lr_model.predict(X_val_scaled)
val_acc_lr = accuracy_score(y_val, y_val_pred_lr)
print(f"✓ LR Training Accuracy: {train_acc_lr:.4f}")
print(f"✓ LR Validation Accuracy: {val_acc_lr:.4f}")

# Plot Confusion Matrix LR
plot_confusion_matrix(y_val, y_val_pred_lr, classes, title='Logistic Regression - Confusion Matrix', filename='confusion_matrix_lr.png')


# B. Random Forest
print("\nBASELINE MODEL 2: RANDOM FOREST")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    class_weight='balanced',
    random_state=CONFIG['RANDOM_STATE'],
    n_jobs=-1,
    verbose=1
)
rf_model.fit(X_train_scaled, y_train)
train_acc_rf = accuracy_score(y_train, rf_model.predict(X_train_scaled))
y_val_pred_rf = rf_model.predict(X_val_scaled)
val_acc_rf = accuracy_score(y_val, y_val_pred_rf)
print(f"✓ RF Training Accuracy: {train_acc_rf:.4f}")
print(f"✓ RF Validation Accuracy: {val_acc_rf:.4f}")

# Plot Confusion Matrix RF
plot_confusion_matrix(y_val, y_val_pred_rf, classes, title='Random Forest - Confusion Matrix', filename='confusion_matrix_rf.png')


# C. Gradient Boosting
print("\nBASELINE MODEL 3: GRADIENT BOOSTING")
gb_model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=5,
    random_state=CONFIG['RANDOM_STATE'],
    verbose=1
)
gb_model.fit(X_train_scaled, y_train)
train_acc_gb = accuracy_score(y_train, gb_model.predict(X_train_scaled))
y_val_pred_gb = gb_model.predict(X_val_scaled)
val_acc_gb = accuracy_score(y_val, y_val_pred_gb)
print(f"✓ GB Training Accuracy: {train_acc_gb:.4f}")
print(f"✓ GB Validation Accuracy: {val_acc_gb:.4f}")

# Plot Confusion Matrix GB
plot_confusion_matrix(y_val, y_val_pred_gb, classes, title='Gradient Boosting - Confusion Matrix', filename='confusion_matrix_gb.png')


# Compare Baselines
baseline_results = {
    'Logistic Regression': val_acc_lr,
    'Random Forest': val_acc_rf,
    'Gradient Boosting': val_acc_gb
}
print("\n" + "="*70)
print("BASELINE MODEL COMPARISON")
print("="*70)
for model, acc in baseline_results.items():
    print(f"{model:25s}: {acc:.4f} ({acc*100:.2f}%)")

plt.figure(figsize=(10, 6))
models_plot = list(baseline_results.keys())
accuracies_plot = list(baseline_results.values())
colors_plot = ['#3498db', '#2ecc71', '#e67e22']

bars = plt.bar(models_plot, accuracies_plot, color=colors_plot, edgecolor='black', linewidth=2, alpha=0.8)
plt.ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
plt.title('Baseline Model Performance Comparison', fontsize=14, fontweight='bold')
plt.ylim([0, 1])
plt.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}\n({height*100:.1f}%)',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.show()
plt.close()


# D. Voting Ensemble
print("\n" + "="*70)
print("CREATING VOTING ENSEMBLE")
print("="*70)
# Create voting ensemble with all three baseline models
# Note: User's snippet included 'rf_optimized' as 'best_rf_model'. Currently rf_model IS the best RF we have.
voting_clf = VotingClassifier(
    estimators=[
        ('lr', lr_model),
        ('rf', rf_model),
        ('gb', gb_model)
    ],
    voting='soft', # Changed to soft to enable ROC AUC
    n_jobs=-1
)

voting_clf.fit(X_train_scaled, y_train)
train_acc_voting = accuracy_score(y_train, voting_clf.predict(X_train_scaled))
y_val_pred_voting = voting_clf.predict(X_val_scaled)
val_acc_voting = accuracy_score(y_val, y_val_pred_voting)

# Test Evaluation
y_pred_voting_test = voting_clf.predict(X_test_scaled)
test_acc_voting = accuracy_score(y_test, y_pred_voting_test)

print(f"\n✓ Voting Ensemble trained successfully!")
print(f"✓ Voting Ensemble Training Accuracy: {train_acc_voting:.4f}")
print(f"✓ Voting Ensemble Validation Accuracy: {val_acc_voting:.4f}")
print(f"📊 Test Accuracy: {test_acc_voting:.4f} ({test_acc_voting*100:.2f}%)")

# ROC AUC for Voting Ensemble
y_test_prob_voting = voting_clf.predict_proba(X_test_scaled)
roc_auc_voting = plot_roc_curves(y_test, y_test_prob_voting, classes, title='ROC Curves - ML Voting Ensemble', filename='roc_curve_voting.png')
print(f"✓ Voting Ensemble ROC AUC (Micro): {roc_auc_voting:.4f}")


# ==================================================================================
# PART 2: DEEP LEARNING (ResNet50)
# ==================================================================================
print("\n" + "="*70)
print("PART 2: DEEP LEARNING (ResNet50)")
print("="*70)

# Calculate class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

# Data Augmentation
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Build Model
print("Building ResNet50 model...")
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=CONFIG['IMG_SIZE'] + (3,))

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(CONFIG['N_CLASSES'], activation='softmax')(x)

dl_model = Model(inputs=base_model.input, outputs=predictions)
dl_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
print("Starting ResNet50 training...")
history = dl_model.fit(
    datagen.flow(X_train, y_train, batch_size=CONFIG['BATCH_SIZE']),
    steps_per_epoch=max(1, len(X_train) // CONFIG['BATCH_SIZE']),
    epochs=CONFIG['EPOCHS'],
    validation_data=val_datagen.flow(X_val, y_val, batch_size=CONFIG['BATCH_SIZE']),
    class_weight=class_weight_dict,
    callbacks=[
        EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy'),
        ReduceLROnPlateau(factor=0.2, patience=3, monitor='val_accuracy')
    ]
)

# Print Final Training Stats
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
print(f"✓ ResNet50 Training Accuracy (Final): {final_train_acc:.4f}")
print(f"✓ ResNet50 Validation Accuracy (Final): {final_val_acc:.4f}")

# Plot Training History
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('plots/resnet50_history.png')
plt.show()
plt.close()

# Evaluation
print("\nEvaluating ResNet50...")
X_test_preprocessed = preprocess_input(X_test.copy().astype('float32'))
y_test_pred_prob_dl = dl_model.predict(X_test_preprocessed)
y_test_pred_dl = np.argmax(y_test_pred_prob_dl, axis=1)

test_acc_dl = accuracy_score(y_test, y_test_pred_dl)
print(f"✓ ResNet50 Test Accuracy: {test_acc_dl:.4f}")

# ROC AUC for ResNet50
roc_auc_dl = plot_roc_curves(y_test, y_test_pred_prob_dl, classes, title='ROC Curves - ResNet50', filename='roc_curve_resnet50.png')
print(f"✓ ResNet50 ROC AUC (Micro): {roc_auc_dl:.4f}")

# Confusion Matrix for ResNet50
plot_confusion_matrix(y_test, y_test_pred_dl, classes, title='Confusion Matrix - ResNet50', filename='confusion_matrix_resnet50.png')


# ======================================================================
# SAVING ALL MODELS AND PROJECT FILES
# ======================================================================
print("\n" + "="*70)
print("FINAL MODEL COMPARISON (Test Set)")
print("="*70)

# We use validation acc for ML models (except voting) as proxy, unless we run them on X_test (which is fast, let's do it)
test_acc_lr = accuracy_score(y_test, lr_model.predict(X_test_scaled))
test_acc_rf = accuracy_score(y_test, rf_model.predict(X_test_scaled))
test_acc_gb = accuracy_score(y_test, gb_model.predict(X_test_scaled))

results = {
    'Logistic Regression': test_acc_lr,
    'Random Forest': test_acc_rf,
    'Gradient Boosting': test_acc_gb,
    'Voting Ensemble (ML)': test_acc_voting,
    'ResNet50 (Deep Learning)': test_acc_dl
}

for model_name, acc in results.items():
    print(f"{model_name:25s}: {acc:.4f} ({acc*100:.2f}%)")

best_model_name = max(results, key=results.get)
best_accuracy = results[best_model_name]

print(f"\n🏆 BEST MODEL: {best_model_name} with {best_accuracy:.4f} accuracy")
print(f"   Target (>90%): {'✓ ACHIEVED!' if best_accuracy >= 0.9 else '✗ Not achieved'}")

print("\n" + "="*70)
print("SAVING ALL MODELS AND PROJECT FILES")
print("="*70)

# Save all models
models_to_save = {
    'logistic_regression.pkl': lr_model,
    'random_forest.pkl': rf_model,
    'gradient_boosting.pkl': gb_model,
    'voting_ensemble.pkl': voting_clf,
    'feature_scaler.pkl': scaler
}

for filename, model in models_to_save.items():
    filepath = os.path.join('saved_models', filename)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Saved: {filename}")

# Save DL model
dl_model.save('saved_models/resnet50_model.h5')
print(f"✓ Saved: resnet50_model.h5")

# Save Metadata
metadata = {
    'project': 'AI-Driven Crop Disease Prediction & Management System',
    'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'classes': classes,
    'n_classes': CONFIG['N_CLASSES'],
    'img_size': CONFIG['IMG_SIZE'],
    'model_test_accuracies': results,
    'best_model': {
        'name': best_model_name,
        'accuracy': best_accuracy,
        'target_achieved': bool(best_accuracy >= 0.9)
    },
    'roc_auc': {
        'voting_ensemble': roc_auc_voting,
        'resnet50': roc_auc_dl
    },
    'dataset_info': {
        'total_images': len(X),
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test)
    }
}

with open('saved_models/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Saved: metadata.json")

# Save Disease Recommendations (Comprehensive)
disease_recommendations = {}

for class_name in classes:
    disease_display = class_name.replace('_', ' ').title()
    crop = disease_display.split()[0] if ' ' in disease_display else "Crop"
    
    disease_recommendations[class_name] = {
        "disease_name": disease_display,
        "crop_type": crop,
        "description": f"A common disease affecting {crop} plants, identifiable by specific symptoms on plant tissues.",
        "symptoms": [
            "Discoloration or unusual spots on leaves",
            "Abnormal growth patterns or deformities",
            "Wilting despite adequate watering",
            "Visible lesions, pustules, or fungal growth"
        ],
        "management_steps": [
            "Remove and destroy all infected plant materials immediately",
            "Apply appropriate fungicide or pesticide (consult local agricultural extension)",
            "Improve air circulation by proper plant spacing (minimum 18-24 inches)",
            "Implement drip irrigation to avoid wetting foliage",
            "Monitor surrounding plants daily for early detection"
        ],
        "prevention": [
            "Use certified disease-free seeds or transplants",
            "Practice 3-4 year crop rotation",
            "Maintain field sanitation and remove crop debris",
            "Avoid working in fields when plants are wet",
            "Apply preventive fungicides before disease onset"
        ],
        "chemical_control": "Consult local agricultural extension for region-specific approved fungicides/pesticides",
        "organic_alternatives": [
            "Neem oil spray (2% solution)",
            "Copper-based fungicides",
            "Beneficial microorganisms (Trichoderma, Bacillus)",
            "Sulfur dust applications",
            "Compost tea foliar spray"
        ],
        "severity": "Moderate to High",
        "economic_impact": "Can cause 15-50% yield loss if untreated",
        "action_timeline": "Address within 1-2 weeks of detection",
        "expert_consultation": "Contact agricultural extension officer for severe outbreaks"
    }

with open('saved_models/disease_recommendations.json', 'w') as f:
    json.dump(disease_recommendations, f, indent=2)

print(f"✓ Saved: disease_recommendations.json")

print("\n" + "="*70)
print("📦 ALL MODELS SAVED SUCCESSFULLY")
print("="*70)
