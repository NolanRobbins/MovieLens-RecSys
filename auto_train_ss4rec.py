#!/usr/bin/env python3
"""
SS4Rec vs Neural CF Auto-Trainer with Discord Notifications
Enhanced version for MovieLens RecSys project

Usage:
  python auto_train_ss4rec.py --model ncf      # Train Neural CF baseline
  python auto_train_ss4rec.py --model ss4rec   # Train SS4Rec SOTA (when available)
"""

import os
import sys
import subprocess
import requests
import time
import argparse
from datetime import datetime

def send_discord_notification(message: str, webhook_url: str, color: int = 5763719) -> bool:
    """Send completion notification via Discord webhook"""
    try:
        payload = {
            "embeds": [{
                "title": "🎬 MovieLens RecSys Training Update",
                "description": message,
                "color": color,
                "timestamp": datetime.utcnow().isoformat(),
                "footer": {"text": "RunPod A6000 Auto-Trainer"}
            }]
        }
        response = requests.post(webhook_url, json=payload, timeout=10)
        return response.status_code == 204
    except Exception as e:
        print(f"Discord notification failed: {e}")
        return False

def get_system_info() -> str:
    """Get system information"""
    try:
        hostname = subprocess.check_output(['hostname'], text=True).strip()
        gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                                         text=True).strip()
        return f"Instance: {hostname} | GPU: {gpu_info}"
    except:
        return "RunPod A6000 Instance"

def extract_results_from_log(log_file: str, model_type: str) -> str:
    """Extract training results from log file"""
    try:
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        # Look for RMSE results
        import re
        rmse_matches = re.findall(r'Best RMSE: ([\d.]+)', log_content)
        
        if rmse_matches:
            best_rmse = float(rmse_matches[-1])
            
            # Determine performance assessment
            if model_type == 'ncf':
                target_rmse = 0.90
                performance = "✅ Good baseline" if best_rmse <= target_rmse else "⚠️ Above target"
            else:  # ss4rec
                target_rmse = 0.70
                performance = "🏆 SOTA achieved!" if best_rmse <= target_rmse else "📈 Good improvement"
            
            return f"🎯 **Best RMSE: {best_rmse:.4f}** ({performance})"
        
        elif "Training completed successfully" in log_content.lower():
            return "✅ Training completed successfully"
        elif "completed" in log_content.lower():
            return "Training completed (check logs for details)"
        else:
            return "Check logs and W&B dashboard for results"
            
    except Exception as e:
        return f"Could not extract results: {e}"

def main():
    parser = argparse.ArgumentParser(description="Auto-train with Discord notifications")
    parser.add_argument('--model', type=str, choices=['ncf', 'ss4rec'], 
                       default='ncf', help='Model to train')
    parser.add_argument('--config', type=str, default=None,
                       help='Custom config file')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable W&B logging')
    parser.add_argument('--debug', action='store_true',
                       help='Enable comprehensive debug logging for NaN detection')
    
    args = parser.parse_args()
    
    # Project configuration
    project_name = f"MovieLens RecSys - {args.model.upper()}"
    training_script = "runpod_training_wandb.py"
    
    # Discord webhook from environment variable
    webhook_url = os.environ.get('DISCORD_WEBHOOK_URL', '')
    if not webhook_url:
        print("❌ No Discord webhook URL found.")
        print("Set it with: export DISCORD_WEBHOOK_URL='https://discord.com/api/webhooks/...'")
        sys.exit(1)
    
    # Check if training script exists
    if not os.path.exists(training_script):
        print(f"❌ Training script not found: {training_script}")
        sys.exit(1)
    
    # Build command arguments - use virtual environment python
    venv_python = os.path.join('.venv', 'bin', 'python')
    if not os.path.exists(venv_python):
        venv_python = sys.executable  # fallback to system python
        print(f"⚠️  Virtual environment python not found, using: {venv_python}")
    else:
        print(f"🐍 Using virtual environment python: {venv_python}")
    
    cmd_args = [venv_python, training_script, '--model', args.model]
    if args.config:
        cmd_args.extend(['--config', args.config])
    if args.no_wandb:
        cmd_args.append('--no-wandb')
    if args.debug:
        cmd_args.append('--debug')
    
    print(f"""
🎬 MovieLens RecSys Auto-Trainer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚀 Project: {project_name}
📝 Script: {training_script}
🎯 Model: {args.model.upper()}
🔍 Debug Mode: {'ENABLED - Comprehensive NaN detection logging' if args.debug else 'Disabled'}
📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔕 You can close this terminal - training will continue
📱 Discord notification will be sent when complete
""")
    
    # Send start notification
    system_info = get_system_info()
    start_message = f"""
**🚀 Training Started - {args.model.upper()}**
{'🔍 **DEBUG MODE ENABLED** - NaN detection active' if args.debug else ''}

📊 **Model:** {args.model.upper()} ({'Baseline' if args.model == 'ncf' else 'SOTA 2025'})
⏰ **Started:** {datetime.now().strftime('%H:%M:%S UTC')}
🖥️ **System:** {system_info}

{'🎯 **Target:** Validation RMSE < 0.90' if args.model == 'ncf' else '🏆 **Target:** Validation RMSE < 0.70 (SOTA)'}

⏳ Training in progress... notification will be sent when complete.
    """.strip()
    
    send_discord_notification(start_message, webhook_url, color=3447003)  # Blue for start
    
    # Create log file
    log_file = f"training_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Start training in background
    try:
        start_time = time.time()
        
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd_args,
                stdout=f,
                stderr=subprocess.STDOUT
            )
        
        print(f"✅ Training started with PID: {process.pid}")
        print(f"📊 View logs: tail -f {log_file}")
        print("")
        
        # Wait for completion
        process.wait()
        
        # Calculate duration
        duration_hours = (time.time() - start_time) / 3600
        
        # Extract results
        results_info = extract_results_from_log(log_file, args.model)
        
        # Determine success/failure and create message
        if process.returncode == 0:
            # Success
            message = f"""
**🎉 {project_name} - Training Complete!**

⏱️ **Duration:** {duration_hours:.1f} hours
{results_info}
🖥️ **System:** {system_info}

📊 **Check your W&B dashboard for detailed metrics**
💾 **Log file:** `{log_file}`
📈 **Results directory:** `results/{args.model}_*`

✅ **Instance can now be safely terminated.**
            """.strip()
            
            color = 5763719  # Green for success
            
        else:
            # Failure
            message = f"""
**❌ {project_name} - Training Failed**

⏱️ **Duration:** {duration_hours:.1f} hours
❌ **Exit Code:** {process.returncode}
🖥️ **System:** {system_info}

📝 **Check log file for details:** `{log_file}`
🔧 **May need to restart training**

Common fixes:
• Check GPU memory: `nvidia-smi`
• Verify data files: `ls data/processed/`
• Check config: `configs/{args.model}*.yaml`
            """.strip()
            
            color = 15158332  # Red for failure
        
        # Send completion notification
        if send_discord_notification(message, webhook_url, color):
            print("✅ Discord notification sent successfully!")
        else:
            print("❌ Failed to send Discord notification")
            print(f"Results: {message}")
        
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
        
        interrupt_message = f"""
**🛑 {project_name} - Training Interrupted**

⏱️ **Duration:** {(time.time() - start_time) / 3600:.1f} hours
👤 **Reason:** User interruption
🖥️ **System:** {system_info}

🔄 **Training can be resumed if needed**
        """.strip()
        
        send_discord_notification(interrupt_message, webhook_url, color=16776960)  # Yellow for warning
        sys.exit(1)
        
    except Exception as e:
        error_message = f"""
**💥 {project_name} - Training Error**

❌ **Error:** {str(e)}
🖥️ **System:** {system_info}

🔧 **Check system setup and try again**
        """.strip()
        
        print(f"❌ Failed to start training: {e}")
        send_discord_notification(error_message, webhook_url, color=15158332)  # Red for error
        sys.exit(1)

if __name__ == "__main__":
    main()