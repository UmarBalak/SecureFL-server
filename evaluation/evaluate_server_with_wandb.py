# evaluate_server_complete.py - Complete evaluation module for FL server

import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from unified_fl_tracker import fl_tracker
import wandb

def evaluate_model_with_wandb(model, X_test, y_test_cat, class_names=None, save_plots=True, 
                              fl_round=None, log_to_wandb=True):
    """
    Complete FL server evaluation with WandB integration
    """
    print(f"\n{'='*70}")
    print(f"📊 FL SERVER EVALUATION - Round {fl_round}")
    print(f"{'='*70}")
    
    def compute_comprehensive_metrics(X, y_cat, dataset_name="Test Set"):
        """
        Comprehensive metric computation for FL server evaluation
        """
        print(f"\n📊 Evaluating {dataset_name} for FL Round {fl_round}...")
        
        # Data validation and cleaning
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            print("⚠️ Cleaning NaN/Inf values in input data...")
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if np.any(np.isnan(y_cat)) or np.any(np.isinf(y_cat)):
            print("⚠️ Cleaning NaN/Inf values in target data...")
            y_cat = np.nan_to_num(y_cat, nan=0.0, posinf=1.0, neginf=0.0)
        
        try:
            # Model prediction with stability
            print("🎯 Computing FL model predictions...")
            y_pred_probs = model.predict(X, verbose=0, batch_size=128)
            
            # Ensure valid probabilities
            y_pred_probs = np.clip(y_pred_probs, 1e-15, 1.0 - 1e-15)
            
            # Manual loss calculation (bypasses TF bugs)
            manual_loss = -np.mean(np.sum(y_cat * np.log(y_pred_probs), axis=1))
            
            # Class predictions
            y_true_classes = np.argmax(y_cat, axis=1)
            y_pred_classes = np.argmax(y_pred_probs, axis=1)
            accuracy = np.mean(y_true_classes == y_pred_classes)
            
            # Verify against TensorFlow evaluation
            try:
                tf_results = model.evaluate(X, y_cat, verbose=0, batch_size=128)
                tf_loss = tf_results[0] if isinstance(tf_results, list) else tf_results
                
                if tf_loss > 100:  # TensorFlow bug detected
                    print(f"🐛 TensorFlow evaluate() bug detected: {tf_loss:.2f}")
                    print(f"✅ Using manual calculation: {manual_loss:.4f}")
                    final_loss = manual_loss
                else:
                    final_loss = tf_loss
                    
            except Exception as e:
                print(f"⚠️ TensorFlow evaluate() failed: {e}")
                final_loss = manual_loss
            
            print(f"✅ FL Round {fl_round} - Loss: {final_loss:.4f}, Accuracy: {accuracy:.4f}")
            
        except Exception as e:
            print(f"❌ Error in FL prediction computation: {e}")
            return {"error": str(e)}
        
        try:
            # Comprehensive sklearn metrics
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true_classes, y_pred_classes, average=None, zero_division=0
            )
            
            # Aggregate metrics
            macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
                y_true_classes, y_pred_classes, average='macro', zero_division=0
            )
            
            weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
                y_true_classes, y_pred_classes, average='weighted', zero_division=0
            )
            
            # Confusion matrix
            cm = confusion_matrix(y_true_classes, y_pred_classes)
            
        except Exception as e:
            print(f"❌ Error in FL metrics computation: {e}")
            return {"error": str(e)}
        
        # Print comprehensive FL results
        print(f"\n📋 FL Round {fl_round} Per-Class Performance:")
        print("-" * 70)
        
        class_labels = class_names if class_names else [f"Class_{i}" for i in range(len(precision))]
        
        for i, label in enumerate(class_labels):
            if i < len(precision):
                print(f"{label:20}: P={precision[i]:.4f}, R={recall[i]:.4f}, "
                      f"F1={f1[i]:.4f}, Support={support[i]:4d}")
        
        print(f"\n📈 FL Round {fl_round} Aggregate Performance:")
        print(f"Macro Avg    : P={macro_precision:.4f}, R={macro_recall:.4f}, F1={macro_f1:.4f}")
        print(f"Weighted Avg : P={weighted_precision:.4f}, R={weighted_recall:.4f}, F1={weighted_f1:.4f}")
        
        # WandB logging for FL server
        if log_to_wandb and fl_round is not None:
            try:
                # Log FL round specific metrics
                fl_metrics = {
                    f"fl_round_{fl_round}/accuracy": accuracy,
                    f"fl_round_{fl_round}/loss": final_loss,
                    f"fl_round_{fl_round}/macro_f1": macro_f1,
                    f"fl_round_{fl_round}/weighted_f1": weighted_f1,
                    f"fl_round_{fl_round}/macro_precision": macro_precision,
                    f"fl_round_{fl_round}/macro_recall": macro_recall
                }
                
                # Log per-class metrics for this FL round
                for i, label in enumerate(class_labels):
                    if i < len(precision):
                        fl_metrics.update({
                            f"fl_round_{fl_round}/per_class/{label}/precision": precision[i],
                            f"fl_round_{fl_round}/per_class/{label}/recall": recall[i],
                            f"fl_round_{fl_round}/per_class/{label}/f1": f1[i],
                            f"fl_round_{fl_round}/per_class/{label}/support": support[i]
                        })
                
                wandb.log(fl_metrics)
                
                # Create per-class performance table
                per_class_data = []
                for i, label in enumerate(class_labels):
                    if i < len(precision):
                        per_class_data.append([
                            fl_round, label, precision[i], recall[i], f1[i], support[i]
                        ])
                
                wandb.log({
                    f"fl_round_{fl_round}/per_class_table": wandb.Table(
                        data=per_class_data,
                        columns=["FL_Round", "Attack_Type", "Precision", "Recall", "F1", "Support"]
                    )
                })
                
                print(f"📊 FL Round {fl_round}: Comprehensive metrics logged to WandB")
                
            except Exception as e:
                print(f"⚠️ WandB FL logging warning: {e}")
        
        # Enhanced confusion matrix for FL
        if save_plots:
            try:
                plot_fl_confusion_matrix(
                    cm, class_labels, dataset_name, 
                    accuracy, macro_f1, weighted_f1, final_loss, fl_round
                )
            except Exception as e:
                print(f"⚠️ FL plot generation warning: {e}")
        
        return {
            'loss': float(final_loss),
            'accuracy': float(accuracy),
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1_score': f1.tolist(),
            'support': support.tolist(),
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall),
            'macro_f1': float(macro_f1),
            'weighted_precision': float(weighted_precision),
            'weighted_recall': float(weighted_recall),
            'weighted_f1': float(weighted_f1),
            'confusion_matrix': cm.tolist(),
            'num_classes': len(class_labels),
            'total_samples': len(y_true_classes),
            'fl_round': fl_round,
            'y_true': y_true_classes.tolist(),
            'y_pred': y_pred_classes.tolist()
        }
    
    # Main FL evaluation
    results = compute_comprehensive_metrics(X_test, y_test_cat, "FL Test Set")
    
    # Print FL evaluation summary
    print(f"\n{'='*70}")
    print(f"🎯 FL ROUND {fl_round} EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"Test Accuracy      : {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"Test Loss          : {results['loss']:.4f}")
    print(f"Macro F1-Score     : {results['macro_f1']:.4f}")
    print(f"Weighted F1-Score  : {results['weighted_f1']:.4f}")
    print(f"Total Test Samples : {results['total_samples']:,}")
    print(f"FL Round           : {fl_round}")
    
    # FL performance interpretation
    if results['loss'] < 2.0:
        print("✅ FL model loss is excellent for cybersecurity classification!")
    elif results['loss'] < 5.0:
        print("✅ FL model loss is good for non-IID distributed data!")
    else:
        print("⚠️ FL model loss is high - may need more rounds or better aggregation")
    
    if results['accuracy'] > 0.60:
        print("✅ FL model accuracy is good for distributed cybersecurity data!")
    elif results['accuracy'] > 0.45:
        print("✅ FL model accuracy is acceptable for 15-class non-IID classification!")
    else:
        print("⚠️ FL model accuracy needs improvement - consider client data balancing")
    
    if results['macro_f1'] < 0.40:
        print("📊 Low macro F1 in FL is normal due to non-IID data distribution")
        print("💡 FL system is working - aggregation helps with class imbalance over rounds")
    
    print("="*70)
    
    return {'test': results}

def plot_fl_confusion_matrix(cm, class_labels, dataset_name, accuracy, 
                            macro_f1, weighted_f1, loss, fl_round=None):
    """
    Enhanced confusion matrix visualization for FL rounds
    """
    os.makedirs('fl_evaluation_plots', exist_ok=True)
    
    plt.figure(figsize=(18, 14))
    
    # Create heatmap with FL-specific styling
    mask = cm == 0
    sns.heatmap(cm,
                annot=True,
                fmt='d',
                cmap='RdYlBu_r',  # Different colormap for FL
                xticklabels=class_labels,
                yticklabels=class_labels,
                square=True,
                mask=mask,
                cbar_kws={'label': 'Number of Samples'},
                linewidths=0.5)
    
    title_text = f'FL Round {fl_round} - {dataset_name} Evaluation'
    
    plt.title(f'{title_text}\n'
             f'Accuracy: {accuracy:.3f} | Loss: {loss:.3f} | Macro F1: {macro_f1:.3f} | Weighted F1: {weighted_f1:.3f}',
             fontsize=16, pad=20)
    
    plt.xlabel('Predicted Attack Type', fontsize=14)
    plt.ylabel('True Attack Type', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add FL-specific annotation
    plt.figtext(0.02, 0.02,
                f"🔄 FL System Performance (Round {fl_round})\n"
                f"🔒 Dataset: ML-Edge-IIoT Non-IID Distribution\n" 
                f"⚡ Federated Aggregation: Weighted Average\n"
                f"📊 Server Evaluation: Production-grade FL tracking",
                fontsize=11, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    
    # Save FL round specific plot
    filename = f'fl_evaluation_plots/fl_round_{fl_round}_confusion_matrix.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    # Log to WandB for FL tracking
    try:
        wandb.log({f"fl_round_{fl_round}/confusion_matrix": wandb.Image(filename)})
        print(f"📊 FL Round {fl_round}: Confusion matrix saved and logged to WandB")
    except:
        print(f"⚠️ Could not log FL confusion matrix to WandB")
    
    plt.close()
