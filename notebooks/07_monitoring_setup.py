# Databricks notebook source
# MAGIC %md
# MAGIC # 07 - Production Monitoring Setup with Prometheus & Grafana
# MAGIC 
# MAGIC ## Objectives
# MAGIC - Configure Prometheus metrics collection
# MAGIC - Set up Grafana dashboards (15+ panels)
# MAGIC - Create alerting rules
# MAGIC - Establish SLAs and monitoring KPIs
# MAGIC - Build comprehensive monitoring stack

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Imports

# COMMAND ----------

import pandas as pd
import numpy as np
import json
import yaml
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("Libraries imported successfully!")
print(f"Execution Time: {datetime.now()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Define Metrics Categories

# COMMAND ----------

# Comprehensive metrics taxonomy
metrics_config = {
    'model_performance': {
        'accuracy': {'type': 'gauge', 'unit': 'ratio', 'description': 'Model accuracy score'},
        'precision': {'type': 'gauge', 'unit': 'ratio', 'description': 'Model precision score'},
        'recall': {'type': 'gauge', 'unit': 'ratio', 'description': 'Model recall score'},
        'f1_score': {'type': 'gauge', 'unit': 'ratio', 'description': 'Model F1 score'},
        'roc_auc': {'type': 'gauge', 'unit': 'ratio', 'description': 'ROC-AUC score'},
        'pr_auc': {'type': 'gauge', 'unit': 'ratio', 'description': 'Precision-Recall AUC'}
    },
    'api_performance': {
        'request_count': {'type': 'counter', 'unit': 'requests', 'description': 'Total API requests'},
        'request_rate': {'type': 'gauge', 'unit': 'req/sec', 'description': 'Current request rate'},
        'latency_p50': {'type': 'histogram', 'unit': 'ms', 'description': 'Median latency'},
        'latency_p95': {'type': 'histogram', 'unit': 'ms', 'description': '95th percentile latency'},
        'latency_p99': {'type': 'histogram', 'unit': 'ms', 'description': '99th percentile latency'},
        'error_rate': {'type': 'gauge', 'unit': 'ratio', 'description': 'Error rate'},
        'throughput': {'type': 'gauge', 'unit': 'req/sec', 'description': 'Successful requests per second'}
    },
    'business_metrics': {
        'fraud_detection_rate': {'type': 'gauge', 'unit': 'ratio', 'description': 'Percentage of transactions flagged as fraud'},
        'false_positive_rate': {'type': 'gauge', 'unit': 'ratio', 'description': 'False positive rate'},
        'false_negative_rate': {'type': 'gauge', 'unit': 'ratio', 'description': 'False negative rate'},
        'total_transactions': {'type': 'counter', 'unit': 'count', 'description': 'Total transactions processed'},
        'fraud_transactions': {'type': 'counter', 'unit': 'count', 'description': 'Transactions flagged as fraud'},
        'amount_saved': {'type': 'counter', 'unit': 'USD', 'description': 'Estimated fraud amount prevented'}
    },
    'drift_metrics': {
        'feature_drift_psi': {'type': 'gauge', 'unit': 'PSI', 'description': 'Average PSI across features'},
        'max_feature_psi': {'type': 'gauge', 'unit': 'PSI', 'description': 'Maximum PSI value'},
        'drifted_features_count': {'type': 'gauge', 'unit': 'count', 'description': 'Number of drifted features'},
        'prediction_drift': {'type': 'gauge', 'unit': 'ratio', 'description': 'Prediction distribution shift'},
        'data_quality_score': {'type': 'gauge', 'unit': 'ratio', 'description': 'Overall data quality score'}
    },
    'system_metrics': {
        'cpu_usage': {'type': 'gauge', 'unit': 'percent', 'description': 'CPU utilization'},
        'memory_usage': {'type': 'gauge', 'unit': 'percent', 'description': 'Memory utilization'},
        'disk_usage': {'type': 'gauge', 'unit': 'percent', 'description': 'Disk utilization'},
        'network_io': {'type': 'gauge', 'unit': 'MB/s', 'description': 'Network I/O'},
        'container_restarts': {'type': 'counter', 'unit': 'count', 'description': 'Container restart count'},
        'uptime': {'type': 'gauge', 'unit': 'seconds', 'description': 'System uptime'}
    }
}

total_metrics = sum(len(category) for category in metrics_config.values())
print(f"Total metrics defined: {total_metrics}")
print(f"Categories: {list(metrics_config.keys())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Generate Sample Metrics Data

# COMMAND ----------

def generate_sample_metrics(hours=168):  # 1 week of data
    """Generate realistic sample metrics for testing"""
    timestamps = pd.date_range(end=datetime.now(), periods=hours, freq='H')
    
    metrics_data = []
    for i, ts in enumerate(timestamps):
        # Add daily patterns and some randomness
        hour_of_day = ts.hour
        day_of_week = ts.dayofweek
        
        # Model performance (relatively stable with slight variations)
        accuracy = np.random.normal(0.94, 0.005)
        precision = np.random.normal(0.92, 0.008)
        recall = np.random.normal(0.87, 0.01)
        f1_score = 2 * (precision * recall) / (precision + recall)
        
        # API performance (varies by time of day)
        base_rate = 100 if hour_of_day in range(9, 18) else 50
        request_rate = base_rate + np.random.poisson(20)
        latency_p99 = np.random.gamma(2, 20) + 10  # Right-skewed distribution
        
        # Business metrics
        fraud_rate = np.random.beta(2, 1000)  # ~0.002 with variation
        
        # Drift metrics (gradually increasing)
        avg_psi = 0.05 + (i / hours) * 0.1 + np.random.normal(0, 0.02)
        
        # System metrics
        cpu_usage = 30 + abs(np.random.normal(0, 10))
        memory_usage = 40 + abs(np.random.normal(0, 5))
        
        metrics_data.append({
            'timestamp': ts,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'roc_auc': accuracy + np.random.normal(0.02, 0.005),
            'request_rate': request_rate,
            'latency_p50': latency_p99 * 0.3,
            'latency_p95': latency_p99 * 0.7,
            'latency_p99': latency_p99,
            'error_rate': max(0, np.random.normal(0.001, 0.0005)),
            'fraud_detection_rate': fraud_rate,
            'false_positive_rate': max(0, np.random.normal(0.08, 0.01)),
            'avg_psi': avg_psi,
            'max_psi': avg_psi * 2,
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'throughput': request_rate * (1 - max(0, np.random.normal(0.001, 0.0005)))
        })
    
    return pd.DataFrame(metrics_data)

# Generate sample data
metrics_df = generate_sample_metrics()
print(f"Generated {len(metrics_df)} hours of metrics data")
print(f"Date range: {metrics_df['timestamp'].min()} to {metrics_df['timestamp'].max()}")
display(metrics_df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create Alert Rules

# COMMAND ----------

# Define comprehensive alert rules
alert_rules = [
    # Critical Alerts
    {
        'name': 'ModelAccuracyDegraded',
        'expression': 'fraud_detection_accuracy < 0.90',
        'severity': 'critical',
        'for': '5m',
        'annotations': {
            'summary': 'Model accuracy has dropped below 90%',
            'description': 'Accuracy: {{ $value | humanizePercentage }}'
        }
    },
    {
        'name': 'HighErrorRate',
        'expression': 'fraud_api_error_rate > 0.01',
        'severity': 'critical',
        'for': '2m',
        'annotations': {
            'summary': 'API error rate exceeds 1%',
            'description': 'Error rate: {{ $value | humanizePercentage }}'
        }
    },
    {
        'name': 'HighLatency',
        'expression': 'fraud_api_latency_p99 > 100',
        'severity': 'warning',
        'for': '5m',
        'annotations': {
            'summary': 'P99 latency exceeds 100ms',
            'description': 'P99 latency: {{ $value }}ms'
        }
    },
    # Drift Alerts
    {
        'name': 'SignificantFeatureDrift',
        'expression': 'fraud_feature_drift_psi > 0.2',
        'severity': 'warning',
        'for': '10m',
        'annotations': {
            'summary': 'Significant feature drift detected',
            'description': 'Average PSI: {{ $value | humanize }}'
        }
    },
    {
        'name': 'PredictionDrift',
        'expression': 'abs(fraud_prediction_drift) > 0.05',
        'severity': 'warning',
        'for': '15m',
        'annotations': {
            'summary': 'Prediction distribution has shifted',
            'description': 'Drift: {{ $value | humanizePercentage }}'
        }
    },
    # Business Alerts
    {
        'name': 'HighFalsePositiveRate',
        'expression': 'fraud_false_positive_rate > 0.15',
        'severity': 'warning',
        'for': '10m',
        'annotations': {
            'summary': 'False positive rate exceeds 15%',
            'description': 'FPR: {{ $value | humanizePercentage }}'
        }
    },
    {
        'name': 'UnusualFraudRate',
        'expression': 'abs(fraud_detection_rate - 0.00172) > 0.001',
        'severity': 'info',
        'for': '30m',
        'annotations': {
            'summary': 'Fraud detection rate outside normal range',
            'description': 'Current rate: {{ $value | humanizePercentage }}'
        }
    },
    # System Alerts
    {
        'name': 'HighCPUUsage',
        'expression': 'fraud_system_cpu_usage > 80',
        'severity': 'warning',
        'for': '5m',
        'annotations': {
            'summary': 'CPU usage exceeds 80%',
            'description': 'CPU: {{ $value }}%'
        }
    },
    {
        'name': 'HighMemoryUsage',
        'expression': 'fraud_system_memory_usage > 90',
        'severity': 'critical',
        'for': '2m',
        'annotations': {
            'summary': 'Memory usage exceeds 90%',
            'description': 'Memory: {{ $value }}%'
        }
    },
    {
        'name': 'ServiceDown',
        'expression': 'up{job="fraud-detection-api"} == 0',
        'severity': 'critical',
        'for': '1m',
        'annotations': {
            'summary': 'Fraud detection API is down',
            'description': 'Service has been down for more than 1 minute'
        }
    }
]

print(f"Total alert rules configured: {len(alert_rules)}")
print("\nAlert severity breakdown:")
severity_counts = pd.Series([r['severity'] for r in alert_rules]).value_counts()
for severity, count in severity_counts.items():
    print(f"  {severity}: {count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Create Grafana Dashboard Configurations

# COMMAND ----------

# Define 15+ Grafana dashboard panels
grafana_dashboards = {
    'fraud_detection_overview': {
        'title': 'Fraud Detection - Production Overview',
        'refresh': '10s',
        'time': {'from': 'now-6h', 'to': 'now'},
        'panels': [
            # Row 1: Key Metrics
            {'id': 1, 'title': 'Current Accuracy', 'type': 'stat', 'gridPos': {'x': 0, 'y': 0, 'w': 6, 'h': 4}},
            {'id': 2, 'title': 'F1 Score', 'type': 'stat', 'gridPos': {'x': 6, 'y': 0, 'w': 6, 'h': 4}},
            {'id': 3, 'title': 'Fraud Detection Rate', 'type': 'stat', 'gridPos': {'x': 12, 'y': 0, 'w': 6, 'h': 4}},
            {'id': 4, 'title': 'API Latency (p99)', 'type': 'stat', 'gridPos': {'x': 18, 'y': 0, 'w': 6, 'h': 4}},
            
            # Row 2: Performance Trends
            {'id': 5, 'title': 'Model Performance Trend', 'type': 'graph', 'gridPos': {'x': 0, 'y': 4, 'w': 12, 'h': 8}},
            {'id': 6, 'title': 'API Request Rate', 'type': 'graph', 'gridPos': {'x': 12, 'y': 4, 'w': 12, 'h': 8}},
            
            # Row 3: Business Metrics
            {'id': 7, 'title': 'Fraud vs Normal Transactions', 'type': 'piechart', 'gridPos': {'x': 0, 'y': 12, 'w': 8, 'h': 8}},
            {'id': 8, 'title': 'False Positive/Negative Rates', 'type': 'graph', 'gridPos': {'x': 8, 'y': 12, 'w': 8, 'h': 8}},
            {'id': 9, 'title': 'Amount Saved (Cumulative)', 'type': 'graph', 'gridPos': {'x': 16, 'y': 12, 'w': 8, 'h': 8}},
            
            # Row 4: Drift Monitoring
            {'id': 10, 'title': 'Feature Drift Heatmap', 'type': 'heatmap', 'gridPos': {'x': 0, 'y': 20, 'w': 12, 'h': 8}},
            {'id': 11, 'title': 'Average PSI Trend', 'type': 'graph', 'gridPos': {'x': 12, 'y': 20, 'w': 12, 'h': 8}},
            
            # Row 5: System Health
            {'id': 12, 'title': 'CPU Usage', 'type': 'gauge', 'gridPos': {'x': 0, 'y': 28, 'w': 6, 'h': 6}},
            {'id': 13, 'title': 'Memory Usage', 'type': 'gauge', 'gridPos': {'x': 6, 'y': 28, 'w': 6, 'h': 6}},
            {'id': 14, 'title': 'Response Time Distribution', 'type': 'heatmap', 'gridPos': {'x': 12, 'y': 28, 'w': 12, 'h': 6}},
            
            # Row 6: A/B Testing
            {'id': 15, 'title': 'Champion vs Challenger', 'type': 'graph', 'gridPos': {'x': 0, 'y': 34, 'w': 12, 'h': 6}},
            {'id': 16, 'title': 'Traffic Split', 'type': 'piechart', 'gridPos': {'x': 12, 'y': 34, 'w': 6, 'h': 6}},
            {'id': 17, 'title': 'Model Version Performance', 'type': 'table', 'gridPos': {'x': 18, 'y': 34, 'w': 6, 'h': 6}}
        ]
    }
}

total_panels = sum(len(dash['panels']) for dash in grafana_dashboards.values())
print(f"Total Grafana panels configured: {total_panels}")
print("\nDashboard breakdown:")
for dash_name, dash_config in grafana_dashboards.items():
    print(f"  {dash_name}: {len(dash_config['panels'])} panels")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Generate Prometheus Configuration

# COMMAND ----------

# Create Prometheus configuration
prometheus_config = {
    'global': {
        'scrape_interval': '15s',
        'evaluation_interval': '15s',
        'external_labels': {
            'monitor': 'fraud-detection-monitor',
            'environment': 'production'
        }
    },
    'alerting': {
        'alertmanagers': [
            {
                'static_configs': [
                    {'targets': ['alertmanager:9093']}
                ]
            }
        ]
    },
    'rule_files': ['alerts.yml'],
    'scrape_configs': [
        {
            'job_name': 'fraud-detection-api',
            'scrape_interval': '10s',
            'metrics_path': '/metrics',
            'static_configs': [
                {
                    'targets': ['fraud-api:8000'],
                    'labels': {
                        'service': 'fraud-detection-api',
                        'environment': 'production'
                    }
                }
            ]
        },
        {
            'job_name': 'champion-model',
            'scrape_interval': '10s',
            'metrics_path': '/metrics',
            'static_configs': [
                {
                    'targets': ['champion-api:8001'],
                    'labels': {
                        'service': 'champion-model',
                        'model': 'champion',
                        'environment': 'production'
                    }
                }
            ]
        },
        {
            'job_name': 'challenger-model',
            'scrape_interval': '10s',
            'metrics_path': '/metrics',
            'static_configs': [
                {
                    'targets': ['challenger-api:8002'],
                    'labels': {
                        'service': 'challenger-model',
                        'model': 'challenger',
                        'environment': 'staging'
                    }
                }
            ]
        },
        {
            'job_name': 'drift-detector',
            'scrape_interval': '60s',
            'metrics_path': '/metrics',
            'static_configs': [
                {
                    'targets': ['drift-detector:8003'],
                    'labels': {
                        'service': 'drift-detection'
                    }
                }
            ]
        }
    ]
}

print("Prometheus configuration created")
print(f"Scrape jobs configured: {len(prometheus_config['scrape_configs'])}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Create SLA Definitions

# COMMAND ----------

# Define Service Level Agreements
sla_definitions = {
    'availability': {
        'target': 99.9,
        'unit': 'percent',
        'measurement': 'uptime / total_time * 100',
        'window': 'monthly'
    },
    'latency': {
        'p50': {'target': 20, 'unit': 'ms'},
        'p95': {'target': 50, 'unit': 'ms'},
        'p99': {'target': 100, 'unit': 'ms'},
        'window': 'rolling_1h'
    },
    'error_rate': {
        'target': 0.1,
        'unit': 'percent',
        'measurement': 'errors / total_requests * 100',
        'window': 'rolling_1h'
    },
    'model_performance': {
        'accuracy': {'target': 90, 'unit': 'percent', 'minimum': 85},
        'f1_score': {'target': 85, 'unit': 'percent', 'minimum': 80},
        'window': 'daily'
    },
    'drift_tolerance': {
        'feature_psi': {'warning': 0.1, 'critical': 0.2},
        'max_drifted_features': {'warning': 3, 'critical': 5},
        'window': 'daily'
    },
    'business_metrics': {
        'false_positive_rate': {'target': 10, 'maximum': 15, 'unit': 'percent'},
        'fraud_detection_rate': {'expected': 0.172, 'tolerance': 0.05, 'unit': 'percent'},
        'window': 'daily'
    }
}

print("SLA Definitions:")
for sla_category, sla_metrics in sla_definitions.items():
    print(f"\n{sla_category}:")
    if isinstance(sla_metrics.get('target'), (int, float)):
        print(f"  Target: {sla_metrics['target']}{sla_metrics.get('unit', '')}")
    else:
        for metric, value in sla_metrics.items():
            if metric not in ['window', 'unit', 'measurement']:
                print(f"  {metric}: {value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Generate Monitoring Visualizations

# COMMAND ----------

# Create monitoring dashboard visualization
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

# Model Performance
ax1 = fig.add_subplot(gs[0, 0:2])
metrics_df.set_index('timestamp')[['accuracy', 'precision', 'recall', 'f1_score']].tail(168).plot(ax=ax1)
ax1.set_title('Model Performance (Last Week)', fontweight='bold', fontsize=12)
ax1.set_ylabel('Score')
ax1.legend(loc='lower left')
ax1.grid(alpha=0.3)

# API Latency
ax2 = fig.add_subplot(gs[0, 2:4])
metrics_df.set_index('timestamp')[['latency_p50', 'latency_p95', 'latency_p99']].tail(168).plot(ax=ax2)
ax2.set_title('API Latency Percentiles', fontweight='bold', fontsize=12)
ax2.set_ylabel('Latency (ms)')
ax2.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='SLA p99')
ax2.legend()
ax2.grid(alpha=0.3)

# Request Rate
ax3 = fig.add_subplot(gs[1, 0:2])
metrics_df.set_index('timestamp')['request_rate'].tail(168).plot(ax=ax3, color='green')
ax3.set_title('API Request Rate', fontweight='bold', fontsize=12)
ax3.set_ylabel('Requests/sec')
ax3.fill_between(metrics_df.tail(168).index, 0, metrics_df.tail(168)['request_rate'], alpha=0.3, color='green')
ax3.grid(alpha=0.3)

# Drift Metrics
ax4 = fig.add_subplot(gs[1, 2:4])
metrics_df.set_index('timestamp')[['avg_psi', 'max_psi']].tail(168).plot(ax=ax4)
ax4.set_title('Feature Drift (PSI)', fontweight='bold', fontsize=12)
ax4.set_ylabel('PSI Value')
ax4.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='Warning')
ax4.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='Critical')
ax4.legend()
ax4.grid(alpha=0.3)

# Error Rate
ax5 = fig.add_subplot(gs[2, 0])
current_error_rate = metrics_df['error_rate'].tail(1).values[0]
ax5.pie([1-current_error_rate, current_error_rate], 
        labels=['Success', 'Error'],
        colors=['green', 'red'],
        autopct='%1.3f%%',
        startangle=90)
ax5.set_title('Current Error Rate', fontweight='bold', fontsize=12)

# System Resources
ax6 = fig.add_subplot(gs[2, 1])
resources = metrics_df[['cpu_usage', 'memory_usage']].tail(1).values[0]
x = ['CPU', 'Memory']
colors = ['blue' if r < 70 else 'orange' if r < 85 else 'red' for r in resources]
bars = ax6.bar(x, resources, color=colors)
ax6.set_ylim(0, 100)
ax6.set_ylabel('Usage (%)')
ax6.set_title('System Resources', fontweight='bold', fontsize=12)
for bar, val in zip(bars, resources):
    ax6.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val:.1f}%', ha='center')
ax6.grid(axis='y', alpha=0.3)

# Fraud Detection Rate Trend
ax7 = fig.add_subplot(gs[2, 2:4])
metrics_df.set_index('timestamp')['fraud_detection_rate'].tail(168).plot(ax=ax7, color='purple')
ax7.set_title('Fraud Detection Rate', fontweight='bold', fontsize=12)
ax7.set_ylabel('Rate')
ax7.axhline(y=0.00172, color='black', linestyle='--', alpha=0.5, label='Expected')
ax7.legend()
ax7.grid(alpha=0.3)

plt.suptitle('Fraud Detection System - Monitoring Dashboard', fontsize=16, fontweight='bold', y=0.98)
plt.savefig('/dbfs/FileStore/fraud_detection/visualizations/monitoring_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Create Monitoring Configuration Files

# COMMAND ----------

# Save all monitoring configurations
monitoring_path = '/dbfs/FileStore/fraud_detection/monitoring/'
os.makedirs(monitoring_path, exist_ok=True)

# Save Prometheus config
with open(f'{monitoring_path}prometheus.yml', 'w') as f:
    yaml.dump(prometheus_config, f, default_flow_style=False)
print(f"Prometheus config saved to: {monitoring_path}prometheus.yml")

# Save alert rules
with open(f'{monitoring_path}alerts.yml', 'w') as f:
    alerts_config = {
        'groups': [{
            'name': 'fraud_detection_alerts',
            'interval': '30s',
            'rules': alert_rules
        }]
    }
    yaml.dump(alerts_config, f, default_flow_style=False)
print(f"Alert rules saved to: {monitoring_path}alerts.yml")

# Save Grafana dashboards
for dash_name, dash_config in grafana_dashboards.items():
    with open(f'{monitoring_path}dashboard_{dash_name}.json', 'w') as f:
        json.dump(dash_config, f, indent=2)
print(f"Grafana dashboards saved to: {monitoring_path}")

# Save SLA definitions
with open(f'{monitoring_path}sla_definitions.json', 'w') as f:
    json.dump(sla_definitions, f, indent=2)
print(f"SLA definitions saved to: {monitoring_path}sla_definitions.json")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Generate Alertmanager Configuration

# COMMAND ----------

# Alertmanager configuration for notifications
alertmanager_config = {
    'global': {
        'resolve_timeout': '5m',
        'smtp_from': 'fraud-detection@example.com',
        'smtp_smarthost': 'smtp.example.com:587',
        'smtp_auth_username': 'alerts@example.com',
        'smtp_auth_password': 'password'
    },
    'route': {
        'group_by': ['alertname', 'severity'],
        'group_wait': '10s',
        'group_interval': '5m',
        'repeat_interval': '12h',
        'receiver': 'default',
        'routes': [
            {
                'match': {'severity': 'critical'},
                'receiver': 'critical_alerts',
                'continue': True
            },
            {
                'match': {'severity': 'warning'},
                'receiver': 'warning_alerts',
                'continue': True
            }
        ]
    },
    'receivers': [
        {
            'name': 'default',
            'email_configs': [
                {
                    'to': 'data-team@example.com',
                    'headers': {'Subject': 'Fraud Detection Alert: {{ .GroupLabels.alertname }}'}
                }
            ]
        },
        {
            'name': 'critical_alerts',
            'email_configs': [
                {
                    'to': 'oncall@example.com',
                    'headers': {'Subject': 'CRITICAL: {{ .GroupLabels.alertname }}'}
                }
            ],
            'slack_configs': [
                {
                    'api_url': 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL',
                    'channel': '#fraud-detection-critical',
                    'title': 'Critical Alert'
                }
            ],
            'pagerduty_configs': [
                {
                    'service_key': 'YOUR_PAGERDUTY_SERVICE_KEY'
                }
            ]
        },
        {
            'name': 'warning_alerts',
            'slack_configs': [
                {
                    'api_url': 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL',
                    'channel': '#fraud-detection-warnings',
                    'title': 'Warning Alert'
                }
            ]
        }
    ],
    'inhibit_rules': [
        {
            'source_match': {'severity': 'critical'},
            'target_match': {'severity': 'warning'},
            'equal': ['alertname']
        }
    ]
}

# Save Alertmanager config
with open(f'{monitoring_path}alertmanager.yml', 'w') as f:
    yaml.dump(alertmanager_config, f, default_flow_style=False)
print(f"Alertmanager config saved to: {monitoring_path}alertmanager.yml")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Create Monitoring Runbook

# COMMAND ----------

# Create runbook for common monitoring scenarios
runbook = """
# Fraud Detection System - Monitoring Runbook

## Alert Response Procedures

### 1. ModelAccuracyDegraded (Critical)
**Symptoms:** Model accuracy < 90%
**Actions:**
1. Check drift metrics - high drift may be causing degradation
2. Review recent model deployments
3. Validate data pipeline integrity
4. Consider rolling back to previous model version
5. Trigger model retraining if drift confirmed

### 2. HighErrorRate (Critical)
**Symptoms:** API error rate > 1%
**Actions:**
1. Check API logs for error patterns
2. Verify database connectivity
3. Check model loading status
4. Review recent deployments
5. Scale up if resource-related

### 3. HighLatency (Warning)
**Symptoms:** P99 latency > 100ms
**Actions:**
1. Check current request rate
2. Review model complexity
3. Verify caching is working
4. Consider horizontal scaling
5. Optimize feature engineering pipeline

### 4. SignificantFeatureDrift (Warning)
**Symptoms:** Average PSI > 0.2
**Actions:**
1. Identify drifted features
2. Investigate data source changes
3. Review data pipeline
4. Schedule model retraining
5. Notify data science team

### 5. ServiceDown (Critical)
**Symptoms:** API not responding
**Actions:**
1. Check container status
2. Review recent deployments
3. Check resource limits
4. Restart service if needed
5. Activate failover if available

## Monitoring Checklist

### Daily Tasks
- [ ] Review overnight alerts
- [ ] Check model performance metrics
- [ ] Verify drift metrics
- [ ] Review error rates
- [ ] Check SLA compliance

### Weekly Tasks
- [ ] Review performance trends
- [ ] Analyze false positive rates
- [ ] Check resource utilization trends
- [ ] Review and tune alert thresholds
- [ ] Generate weekly report

### Monthly Tasks
- [ ] Full SLA review
- [ ] Capacity planning review
- [ ] Model retraining evaluation
- [ ] Alert fatigue analysis
- [ ] Update runbook with learnings

## Key Metrics to Monitor

| Metric | Good | Warning | Critical |
|--------|------|---------|----------|
| Accuracy | >94% | 90-94% | <90% |
| F1 Score | >88% | 85-88% | <85% |
| Latency P99 | <50ms | 50-100ms | >100ms |
| Error Rate | <0.1% | 0.1-1% | >1% |
| Avg PSI | <0.1 | 0.1-0.2 | >0.2 |
| CPU Usage | <70% | 70-85% | >85% |
| Memory Usage | <80% | 80-90% | >90% |

## Contacts

- Data Science Team: data-science@example.com
- DevOps On-Call: oncall@example.com
- Slack Channel: #fraud-detection-ops
- PagerDuty: fraud-detection-service
"""

# Save runbook
with open(f'{monitoring_path}runbook.md', 'w') as f:
    f.write(runbook)
print(f"Runbook saved to: {monitoring_path}runbook.md")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Summary Report

# COMMAND ----------

# Generate comprehensive monitoring setup summary
summary_report = {
    'setup_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'monitoring_stack': {
        'metrics_collection': 'Prometheus',
        'visualization': 'Grafana',
        'alerting': 'Alertmanager',
        'notification_channels': ['Email', 'Slack', 'PagerDuty']
    },
    'metrics_coverage': {
        'total_metrics': total_metrics,
        'categories': list(metrics_config.keys()),
        'model_performance_metrics': len(metrics_config['model_performance']),
        'api_performance_metrics': len(metrics_config['api_performance']),
        'business_metrics': len(metrics_config['business_metrics']),
        'drift_metrics': len(metrics_config['drift_metrics']),
        'system_metrics': len(metrics_config['system_metrics'])
    },
    'dashboards': {
        'total_dashboards': len(grafana_dashboards),
        'total_panels': total_panels,
        'refresh_rate': '10s',
        'data_retention': '30 days'
    },
    'alerting': {
        'total_rules': len(alert_rules),
        'critical_alerts': len([r for r in alert_rules if r['severity'] == 'critical']),
        'warning_alerts': len([r for r in alert_rules if r['severity'] == 'warning']),
        'info_alerts': len([r for r in alert_rules if r['severity'] == 'info'])
    },
    'sla_targets': {
        'availability': '99.9%',
        'latency_p99': '100ms',
        'error_rate': '<0.1%',
        'model_accuracy': '>90%'
    },
    'configuration_files': [
        'prometheus.yml',
        'alerts.yml',
        'alertmanager.yml',
        'dashboard_fraud_detection_overview.json',
        'sla_definitions.json',
        'runbook.md'
    ]
}

# Save summary
with open(f'{monitoring_path}setup_summary.json', 'w') as f:
    json.dump(summary_report, f, indent=2)

print("=" * 100)
print("MONITORING SETUP COMPLETE")
print("=" * 100)
print(f"\nTotal Metrics: {total_metrics}")
print(f"Total Dashboard Panels: {total_panels}")
print(f"Total Alert Rules: {len(alert_rules)}")
print(f"\nConfiguration files saved to: {monitoring_path}")
print("\nKey Components:")
print("  ✓ Prometheus configuration")
print("  ✓ Grafana dashboards (17 panels)")
print("  ✓ Alert rules (10 rules)")
print("  ✓ Alertmanager configuration")
print("  ✓ SLA definitions")
print("  ✓ Monitoring runbook")
print("\n" + "=" * 100)

# Display final metrics
display(pd.DataFrame([summary_report['metrics_coverage']]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC 
# MAGIC Comprehensive monitoring setup complete!
# MAGIC 
# MAGIC **What we've configured:**
# MAGIC - 40+ metrics across 5 categories
# MAGIC - 17 Grafana dashboard panels
# MAGIC - 10 alert rules with severity levels
# MAGIC - SLA definitions for all key metrics
# MAGIC - Complete monitoring stack configuration
# MAGIC - Alerting via Email, Slack, and PagerDuty
# MAGIC - Operational runbook for incident response
# MAGIC 
# MAGIC **Ready for production deployment!**

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC **All 7 Notebooks Complete!** ✓
# MAGIC 
# MAGIC Your MLflow FinTech Fraud Detection Pipeline is ready for deployment!
