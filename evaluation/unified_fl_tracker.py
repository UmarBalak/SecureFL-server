# wandb_fl_tracker_synchronized.py - Unified tracker for both Global and Server

import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime
import os

class FederatedLearningTracker:
    """
    Comprehensive WandB tracker for complete FL pipeline: Global Training + Server Aggregation
    """
    
    def __init__(self, project_name="SecureFL-Cybersecurity", entity=None):
        """
        Initialize unified FL tracker for global and server components
        """
        self.project_name = project_name
        self.entity = entity
        self.global_run = None
        self.server_run = None
        self.current_run = None
        self.fl_round = 0
        self.all_metrics = []
        
        # Default config for cybersecurity FL
        self.default_config = {
            "architecture": [256, 256],
            "learning_rate": 5e-5,
            "num_classes": 15,
            "batch_size": 128,
            "dataset": "ML-Edge-IIoT",
            "features": 25,
            "fl_algorithm": "FedAvg",
            "aggregation_strategy": "weighted_average",
            "data_distribution": "non_iid",
            "total_clients": 10,
            "differential_privacy": True,
            "epsilon": 1.0,
            "delta": 1e-5,
        }
    
    def initialize_global_training_run(self, config=None):
        """
        Initialize WandB run for initial global model training
        """
        run_name = f"Global-Training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Merge configs
        final_config = self.default_config.copy()
        if config:
            final_config.update(config)
        final_config.update({"run_type": "global_training", "fl_round": 0})
        
        self.global_run = wandb.init(
            project=self.project_name,
            entity=self.entity,
            name=run_name,
            config=final_config,
            tags=["global_training", "initial_model", "baseline", "cybersecurity"],
            notes="Initial global model training - baseline for FL system",
            reinit="return_previous"
        )
        
        self.current_run = self.global_run
        
        # Define custom metrics for global training
        wandb.define_metric("epoch")
        wandb.define_metric("training/*", step_metric="epoch")
        wandb.define_metric("validation/*", step_metric="epoch")
        wandb.define_metric("evaluation/*", step_metric="epoch")
        
        print(f"🎯 Global Training WandB Run: {run_name}")
        print(f"🔗 URL: {self.global_run.url}")
        return self.global_run
    
    def initialize_server_aggregation_run(self, global_run_id=None, config=None):
        """
        Initialize WandB run for FL server aggregation (linked to global training)
        """
        run_name = f"FL-Server-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Merge configs
        final_config = self.default_config.copy()
        if config:
            final_config.update(config)
        final_config.update({
            "run_type": "server_aggregation",
            "linked_global_run": global_run_id,
            "aggregation_started": True
        })
        
        self.server_run = wandb.init(
            project=self.project_name,
            entity=self.entity,
            name=run_name,
            config=final_config,
            tags=["server_aggregation", "federated_learning", "fl_rounds", "cybersecurity"],
            notes=f"FL Server Aggregation - linked to global run: {global_run_id}",
            reinit="finish_previous"
        )
        
        self.current_run = self.server_run
        self.fl_round = 0
        
        # Define custom metrics for FL rounds
        wandb.define_metric("fl_round")
        wandb.define_metric("global_model/*", step_metric="fl_round")
        wandb.define_metric("per_class/*", step_metric="fl_round")
        wandb.define_metric("client_contribution/*", step_metric="fl_round")
        wandb.define_metric("aggregation/*", step_metric="fl_round")
        
        print(f"🎯 FL Server WandB Run: {run_name}")
        print(f"🔗 URL: {self.server_run.url}")
        if global_run_id:
            print(f"🔗 Linked Global Run: {global_run_id}")
        
        return self.server_run
    
    def log_global_training_metrics(self, epoch, train_metrics, val_metrics=None):
        """
        Log training metrics during initial global model training
        """
        if not self.current_run:
            print("⚠️ No active WandB run for global training")
            return
        
        # Log training metrics
        log_data = {"epoch": epoch}
        
        for metric_name, value in train_metrics.items():
            log_data[f"training/{metric_name}"] = value
        
        if val_metrics:
            for metric_name, value in val_metrics.items():
                log_data[f"validation/{metric_name}"] = value
        
        wandb.log(log_data)
        print(f"📊 Epoch {epoch}: Logged training metrics to WandB")
    
    def log_global_evaluation_results(self, test_metrics, class_names=None):
        """
        Log final evaluation results for global model
        """
        if not self.current_run:
            print("⚠️ No active WandB run")
            return
        
        # Core performance metrics
        eval_metrics = {
            "evaluation/test_accuracy": test_metrics.get("accuracy", 0),
            "evaluation/test_loss": test_metrics.get("loss", float('inf')),
            "evaluation/macro_f1": test_metrics.get("macro_f1", 0),
            "evaluation/weighted_f1": test_metrics.get("weighted_f1", 0),
            "evaluation/macro_precision": test_metrics.get("macro_precision", 0),
            "evaluation/macro_recall": test_metrics.get("macro_recall", 0),
            "evaluation/total_samples": test_metrics.get("total_samples", 0),
            "evaluation/num_classes": test_metrics.get("num_classes", 15)
        }
        
        # Per-class metrics if available
        if "precision" in test_metrics and class_names:
            default_class_names = [
                'Backdoor', 'DDoS_HTTP', 'DDoS_ICMP', 'DDoS_TCP', 'DDoS_UDP',
                'Fingerprinting', 'MITM', 'Normal', 'Password', 'Port_Scanning',
                'Ransomware', 'SQL_injection', 'Uploading', 'Vulnerability_scanner', 'XSS'
            ]
            
            used_class_names = class_names if class_names else default_class_names
            
            for i, class_name in enumerate(used_class_names):
                if i < len(test_metrics.get("precision", [])):
                    eval_metrics.update({
                        f"evaluation/per_class/{class_name}/precision": test_metrics["precision"][i],
                        f"evaluation/per_class/{class_name}/recall": test_metrics["recall"][i],
                        f"evaluation/per_class/{class_name}/f1": test_metrics["f1_score"][i],
                        f"evaluation/per_class/{class_name}/support": test_metrics.get("support", [0])[i]
                    })
        
        wandb.log(eval_metrics)
        
        # Update run summary
        wandb.run.summary.update({
            "final_test_accuracy": test_metrics.get("accuracy", 0),
            "final_test_loss": test_metrics.get("loss", 0),
            "final_macro_f1": test_metrics.get("macro_f1", 0),
            "global_model_ready": True
        })
        
        print(f"📊 Global Evaluation: Logged comprehensive results to WandB")
    
    def log_fl_round_results(self, fl_round, test_metrics, num_contributing_clients, 
                           client_ids=None, model_version=None):
        """
        Log FL round results during server aggregation
        """
        if not self.current_run:
            print("⚠️ No active WandB run for FL aggregation")
            return
        
        self.fl_round = fl_round
        
        # Core FL round metrics
        round_metrics = {
            "fl_round": fl_round,
            "global_model/accuracy": test_metrics.get("accuracy", 0),
            "global_model/loss": test_metrics.get("loss", float('inf')),
            "global_model/macro_f1": test_metrics.get("macro_f1", 0),
            "global_model/weighted_f1": test_metrics.get("weighted_f1", 0),
            "global_model/macro_precision": test_metrics.get("macro_precision", 0),
            "global_model/macro_recall": test_metrics.get("macro_recall", 0),
            "client_contribution/num_clients": num_contributing_clients,
            "client_contribution/participation_rate": num_contributing_clients / self.default_config.get("total_clients", 10),
            "aggregation/timestamp": datetime.now().timestamp()
        }
        
        if model_version:
            round_metrics["global_model/version"] = model_version
        
        # Per-class FL metrics
        if "precision" in test_metrics:
            class_names = [
                'Backdoor', 'DDoS_HTTP', 'DDoS_ICMP', 'DDoS_TCP', 'DDoS_UDP',
                'Fingerprinting', 'MITM', 'Normal', 'Password', 'Port_Scanning',
                'Ransomware', 'SQL_injection', 'Uploading', 'Vulnerability_scanner', 'XSS'
            ]
            
            for i, class_name in enumerate(class_names):
                if i < len(test_metrics["precision"]):
                    round_metrics.update({
                        f"per_class/{class_name}/precision": test_metrics["precision"][i],
                        f"per_class/{class_name}/recall": test_metrics["recall"][i],
                        f"per_class/{class_name}/f1": test_metrics["f1_score"][i],
                        f"per_class/{class_name}/support": test_metrics.get("support", [0])[i]
                    })
        
        # Client participation tracking
        if client_ids:
            round_metrics["client_contribution/active_clients"] = len(set(client_ids))
            client_freq = pd.Series(client_ids).value_counts().to_dict()
            for client_id, freq in client_freq.items():
                round_metrics[f"client_contribution/client_{client_id}"] = freq
        
        wandb.log(round_metrics)
        
        # Store for trend analysis
        metric_record = {
            "fl_round": fl_round,
            "timestamp": datetime.now(),
            **test_metrics,
            "num_contributing_clients": num_contributing_clients
        }
        self.all_metrics.append(metric_record)
        
        print(f"📊 FL Round {fl_round}: Logged to WandB")
        print(f"   Accuracy: {test_metrics.get('accuracy', 0):.4f}")
        print(f"   Macro F1: {test_metrics.get('macro_f1', 0):.4f}")
    
    def create_fl_performance_trends(self):
        """
        Create FL performance trend visualizations
        """
        if len(self.all_metrics) < 2:
            print("⚠️ Need at least 2 FL rounds for trends")
            return None
        
        df = pd.DataFrame(self.all_metrics)
        
        # Create comprehensive FL trends
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("FL Accuracy Progress", "FL Loss Progress", 
                          "F1-Score Evolution", "Client Participation"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Accuracy trend
        fig.add_trace(
            go.Scatter(x=df['fl_round'], y=df['accuracy'], 
                      mode='lines+markers', name='Test Accuracy',
                      line=dict(color='#2E8B57', width=3),
                      marker=dict(size=10)),
            row=1, col=1
        )
        
        # Loss trend
        fig.add_trace(
            go.Scatter(x=df['fl_round'], y=df['loss'],
                      mode='lines+markers', name='Test Loss',
                      line=dict(color='#DC143C', width=3),
                      marker=dict(size=10)),
            row=1, col=2
        )
        
        # F1-Score trends
        fig.add_trace(
            go.Scatter(x=df['fl_round'], y=df['macro_f1'],
                      mode='lines+markers', name='Macro F1',
                      line=dict(color='#4169E1', width=3),
                      marker=dict(size=10)),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['fl_round'], y=df.get('weighted_f1', df['macro_f1']),
                      mode='lines+markers', name='Weighted F1',
                      line=dict(color='#FF8C00', width=3),
                      marker=dict(size=10)),
            row=2, col=1
        )
        
        # Client participation
        fig.add_trace(
            go.Scatter(x=df['fl_round'], y=df['num_contributing_clients'],
                      mode='lines+markers', name='Contributing Clients',
                      line=dict(color='#9370DB', width=3),
                      marker=dict(size=10)),
            row=2, col=2
        )
        
        fig.update_layout(
            title={
                'text': 'Federated Learning Performance Evolution - Cybersecurity Dataset',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            height=800,
            showlegend=True,
            template="plotly_white"
        )
        
        # Update axes
        fig.update_xaxes(title_text="FL Round", row=2, col=1)
        fig.update_xaxes(title_text="FL Round", row=2, col=2)
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=2)
        fig.update_yaxes(title_text="F1-Score", row=2, col=1)
        fig.update_yaxes(title_text="Clients", row=2, col=2)
        
        wandb.log({"fl_performance_trends": wandb.Html(fig.to_html())})
        return fig
    
    def log_model_architecture(self, model_summary_str):
        """
        Log model architecture details
        """
        if self.current_run:
            wandb.log({"model_architecture": wandb.Html(f"<pre>{model_summary_str}</pre>")})
    
    def finalize_run(self, summary_metrics=None):
        """
        Finalize the current WandB run
        """
        if not self.current_run:
            print("⚠️ No active WandB run to finalize")
            return
        
        if summary_metrics:
            for key, value in summary_metrics.items():
                wandb.run.summary[key] = value
        
        # Create final visualizations for FL runs
        if len(self.all_metrics) > 1:
            try:
                self.create_fl_performance_trends()
            except Exception as e:
                print(f"⚠️ Could not create final visualizations: {e}")
        
        print(f"🎯 WandB run finalized: {wandb.run.name}")
        wandb.finish()

# Global tracker instances
fl_tracker = FederatedLearningTracker()
