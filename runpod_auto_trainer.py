#!/usr/bin/env python3
"""
RunPod Auto-Trainer with Background Execution and Notifications
Automatically runs training in background and sends completion notifications
"""

import os
import sys
import time
import subprocess
import json
import requests
from datetime import datetime
from pathlib import Path

# Notification configurations
NOTIFICATION_CONFIG = {
    # Discord webhook (recommended - easy to set up)
    'discord_webhook': os.environ.get('DISCORD_WEBHOOK_URL', ''),
    
    # Slack webhook
    'slack_webhook': os.environ.get('SLACK_WEBHOOK_URL', ''),
    
    # Email via SMTP (Gmail)
    'email': {
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'sender_email': os.environ.get('SENDER_EMAIL', ''),
        'sender_password': os.environ.get('EMAIL_APP_PASSWORD', ''),  # Use app password for Gmail
        'recipient_email': os.environ.get('RECIPIENT_EMAIL', ''),
    },
    
    # Telegram bot
    'telegram': {
        'bot_token': os.environ.get('TELEGRAM_BOT_TOKEN', ''),
        'chat_id': os.environ.get('TELEGRAM_CHAT_ID', ''),
    }
}

def send_discord_notification(message: str, webhook_url: str) -> bool:
    """Send notification via Discord webhook"""
    try:
        payload = {
            "content": message,
            "embeds": [{
                "title": "🚀 RunPod Training Update",
                "description": message,
                "color": 5763719,  # Blue color
                "timestamp": datetime.utcnow().isoformat()
            }]
        }
        response = requests.post(webhook_url, json=payload, timeout=10)
        return response.status_code == 204
    except Exception as e:
        print(f"Discord notification failed: {e}")
        return False

def send_slack_notification(message: str, webhook_url: str) -> bool:
    """Send notification via Slack webhook"""
    try:
        payload = {"text": f"🚀 RunPod Training Update\n{message}"}
        response = requests.post(webhook_url, json=payload, timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"Slack notification failed: {e}")
        return False

def send_email_notification(subject: str, message: str, config: dict) -> bool:
    """Send email notification"""
    try:
        import smtplib
        from email.mime.text import MimeText
        from email.mime.multipart import MimeMultipart
        
        msg = MimeMultipart()
        msg['From'] = config['sender_email']
        msg['To'] = config['recipient_email']
        msg['Subject'] = subject
        
        body = f"""
        🚀 RunPod Training Notification
        
        {message}
        
        Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
        
        Check your W&B dashboard for detailed metrics.
        """
        
        msg.attach(MimeText(body, 'plain'))
        
        server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
        server.starttls()
        server.login(config['sender_email'], config['sender_password'])
        server.send_message(msg)
        server.quit()
        
        return True
    except Exception as e:
        print(f"Email notification failed: {e}")
        return False

def send_telegram_notification(message: str, config: dict) -> bool:
    """Send notification via Telegram bot"""
    try:
        url = f"https://api.telegram.org/bot{config['bot_token']}/sendMessage"
        payload = {
            'chat_id': config['chat_id'],
            'text': f"🚀 RunPod Training Update\n\n{message}",
            'parse_mode': 'Markdown'
        }
        response = requests.post(url, json=payload, timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"Telegram notification failed: {e}")
        return False

def send_notification(message: str) -> None:
    """Send notification via all configured channels"""
    print(f"📢 Sending notification: {message}")
    
    # Discord
    if NOTIFICATION_CONFIG['discord_webhook']:
        if send_discord_notification(message, NOTIFICATION_CONFIG['discord_webhook']):
            print("✅ Discord notification sent")
        else:
            print("❌ Discord notification failed")
    
    # Slack
    if NOTIFICATION_CONFIG['slack_webhook']:
        if send_slack_notification(message, NOTIFICATION_CONFIG['slack_webhook']):
            print("✅ Slack notification sent")
        else:
            print("❌ Slack notification failed")
    
    # Email
    email_config = NOTIFICATION_CONFIG['email']
    if email_config['sender_email'] and email_config['recipient_email']:
        if send_email_notification("RunPod Training Complete", message, email_config):
            print("✅ Email notification sent")
        else:
            print("❌ Email notification failed")
    
    # Telegram
    telegram_config = NOTIFICATION_CONFIG['telegram']
    if telegram_config['bot_token'] and telegram_config['chat_id']:
        if send_telegram_notification(message, telegram_config):
            print("✅ Telegram notification sent")
        else:
            print("❌ Telegram notification failed")

def get_system_info() -> str:
    """Get system information for notifications"""
    try:
        # Get GPU info
        gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], 
                                         text=True).strip()
        # Get instance info
        hostname = subprocess.check_output(['hostname'], text=True).strip()
        
        return f"Instance: {hostname}\nGPU: {gpu_info}"
    except:
        return "System info unavailable"

def monitor_training(log_file: str, process: subprocess.Popen) -> None:
    """Monitor training progress and send notifications"""
    
    # Send start notification
    system_info = get_system_info()
    start_message = f"""
Training Started! 🚀

{system_info}

Log file: {log_file}
Process ID: {process.pid}

Expected duration: 2-3 hours
Target RMSE: 0.52-0.55

You can monitor progress at your W&B dashboard.
    """.strip()
    
    send_notification(start_message)
    
    # Monitor process
    start_time = time.time()
    last_progress_check = 0
    
    while process.poll() is None:
        current_time = time.time()
        elapsed_hours = (current_time - start_time) / 3600
        
        # Send progress update every 30 minutes
        if current_time - last_progress_check > 1800:  # 30 minutes
            progress_message = f"""
Training Progress Update 📊

Elapsed time: {elapsed_hours:.1f} hours
Status: Running
Process ID: {process.pid}

Check W&B dashboard for detailed metrics.
            """.strip()
            
            send_notification(progress_message)
            last_progress_check = current_time
        
        time.sleep(60)  # Check every minute
    
    # Training completed
    total_time = time.time() - start_time
    return_code = process.returncode
    
    if return_code == 0:
        # Success
        try:
            # Try to extract final RMSE from log
            with open(log_file, 'r') as f:
                log_content = f.read()
                # Look for best RMSE in the logs
                best_rmse = "Check logs for final RMSE"
                if "Best RMSE:" in log_content:
                    # Extract the best RMSE value
                    import re
                    rmse_matches = re.findall(r'Best RMSE: ([\d.]+)', log_content)
                    if rmse_matches:
                        best_rmse = f"Best RMSE: {rmse_matches[-1]}"
        except:
            best_rmse = "Check W&B dashboard for final metrics"
        
        completion_message = f"""
🎉 Training Completed Successfully! 

Duration: {total_time/3600:.1f} hours
{best_rmse}

Model saved and uploaded to W&B.
Check your dashboard for complete results and download links.

Instance can now be safely terminated.
        """.strip()
        
    else:
        # Failed
        completion_message = f"""
❌ Training Failed

Duration: {total_time/3600:.1f} hours
Exit code: {return_code}

Check the log file for error details:
{log_file}

You may need to investigate and restart training.
        """.strip()
    
    send_notification(completion_message)

def main():
    """Main auto-trainer function"""
    
    print("🚀 RunPod Auto-Trainer Starting...")
    print("=" * 50)
    
    # Check if training script exists
    training_script = "runpod_training_wandb.py"
    if not os.path.exists(training_script):
        print(f"❌ Training script not found: {training_script}")
        sys.exit(1)
    
    # Set up logging
    log_file = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    print(f"📊 Starting background training...")
    print(f"📝 Log file: {log_file}")
    print(f"🔕 Process will continue if you close this terminal")
    print("")
    
    # Start training process in background
    try:
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                [sys.executable, training_script],
                stdout=f,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True
            )
        
        print(f"✅ Training started with PID: {process.pid}")
        print(f"📈 Monitor progress:")
        print(f"   - Live logs: tail -f {log_file}")
        print(f"   - W&B dashboard: https://wandb.ai/nolanrobbins/movielens-hybrid-vae-a100")
        print("")
        
        # Start monitoring
        monitor_training(log_file, process)
        
    except Exception as e:
        error_message = f"Failed to start training: {e}"
        print(f"❌ {error_message}")
        send_notification(error_message)
        sys.exit(1)

if __name__ == "__main__":
    main()