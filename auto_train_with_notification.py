#!/usr/bin/env python3
"""
Simple Auto-Trainer with Discord Completion Notification
Works for any project - just change the training_script variable
"""

import os
import sys
import subprocess
import requests
import time
from datetime import datetime

def send_discord_notification(message: str, webhook_url: str) -> bool:
    """Send completion notification via Discord webhook"""
    try:
        payload = {
            "embeds": [{
                "title": "🎉 Training Complete!",
                "description": message,
                "color": 5763719,  # Blue for success, red for failure
                "timestamp": datetime.utcnow().isoformat(),
                "footer": {"text": "RunPod Auto-Trainer"}
            }]
        }
        response = requests.post(webhook_url, json=payload, timeout=10)
        return response.status_code == 204
    except Exception as e:
        print(f"Discord notification failed: {e}")
        return False

def get_system_info() -> str:
    """Get basic system info"""
    try:
        hostname = subprocess.check_output(['hostname'], text=True).strip()
        gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                                         text=True).strip()
        return f"Instance: {hostname} | GPU: {gpu_info}"
    except:
        return "RunPod Instance"

def main():
    # Configuration - CHANGE THESE FOR DIFFERENT PROJECTS
    training_script = "runpod_training_wandb.py"  # Change this for different projects
    project_name = "MovieLens RecSys"             # Change this for different projects
    
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
    
    print(f"🚀 Starting background training for: {project_name}")
    print(f"📝 Script: {training_script}")
    print(f"🔕 You can close this terminal - training will continue")
    print(f"📱 Discord notification will be sent when complete")
    print("")
    
    # Create log file
    log_file = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Start training in background
    try:
        start_time = time.time()
        system_info = get_system_info()
        
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                [sys.executable, training_script],
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
        
        # Determine success/failure
        if process.returncode == 0:
            # Success
            try:
                # Try to extract results from log
                with open(log_file, 'r') as f:
                    log_content = f.read()
                    
                # Look for common success indicators
                results_info = "Check W&B dashboard for detailed results"
                if "Best RMSE:" in log_content:
                    import re
                    rmse_matches = re.findall(r'Best RMSE: ([\d.]+)', log_content)
                    if rmse_matches:
                        results_info = f"🎯 Best RMSE: {rmse_matches[-1]}"
                elif "Training completed" in log_content.lower():
                    results_info = "✅ Training completed successfully"
                
            except:
                results_info = "Check logs and W&B dashboard"
            
            message = f"""
**{project_name} - Training Complete! 🎉**

⏱️ **Duration:** {duration_hours:.1f} hours
🎯 **Results:** {results_info}
🖥️ **System:** {system_info}

📊 Check your W&B dashboard for detailed metrics and model downloads.
💾 **Log file:** `{log_file}`

✅ **Instance can now be safely terminated.**
            """.strip()
            
            success_payload = {
                "embeds": [{
                    "title": f"🎉 {project_name} Training Complete!",
                    "description": message,
                    "color": 5763719,  # Blue
                    "timestamp": datetime.utcnow().isoformat()
                }]
            }
            
        else:
            # Failure
            message = f"""
**{project_name} - Training Failed ❌**

⏱️ **Duration:** {duration_hours:.1f} hours
❌ **Exit Code:** {process.returncode}
🖥️ **System:** {system_info}

📝 **Check log file for details:** `{log_file}`
🔧 **May need to restart training**
            """.strip()
            
            success_payload = {
                "embeds": [{
                    "title": f"❌ {project_name} Training Failed",
                    "description": message,
                    "color": 15158332,  # Red
                    "timestamp": datetime.utcnow().isoformat()
                }]
            }
        
        # Send notification
        if send_discord_notification(message, webhook_url):
            print("✅ Discord notification sent successfully!")
        else:
            print("❌ Failed to send Discord notification")
            print(f"Results: {message}")
        
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        error_message = f"❌ Failed to start training: {e}"
        print(error_message)
        send_discord_notification(f"**{project_name} - Training Error**\n\n{error_message}", webhook_url)
        sys.exit(1)

if __name__ == "__main__":
    main()