# 🔔 Discord Auto-Training Setup Guide

## ✅ **Fixed Compatibility Issues**

Your `auto_train_with_notification.py` now works perfectly with our cleaned repository! Here's what was fixed:

### **Problem**
- Old script looked for `runpod_training_wandb.py` ❌
- We cleaned out all old VAE/training files ❌  
- Discord integration was broken ❌

### **Solution**
- ✅ Created new `runpod_training_wandb.py` (compatible with your existing script)
- ✅ Created enhanced `auto_train_ss4rec.py` (better integration)
- ✅ Full Neural CF + SS4Rec support
- ✅ W&B logging maintained

## 🚀 **Quick Start (RunPod)**

### **Option 1: Use Your Existing Script**
```bash
# Your original auto_train_with_notification.py works now!
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/YOUR_WEBHOOK"
python ../auto_train_with_notification.py
```

### **Option 2: Use Enhanced Version**
```bash
# New enhanced version with model selection
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/YOUR_WEBHOOK"

# Train Neural CF baseline (2-3 hours)
python auto_train_ss4rec.py --model ncf

# Train SS4Rec SOTA (4-5 hours, when available)
python auto_train_ss4rec.py --model ss4rec
```

## 📱 **Discord Webhook Setup**

### **1. Create Discord Webhook**
1. Go to Discord server settings
2. Integrations → Webhooks → New Webhook
3. Copy webhook URL
4. Set environment variable:
   ```bash
   export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/YOUR_WEBHOOK_HERE"
   ```

### **2. Test Notification**
```bash
# Quick test (optional)
python -c "
import os, requests
webhook = os.environ.get('DISCORD_WEBHOOK_URL')
if webhook:
    requests.post(webhook, json={'content': '✅ Discord integration test successful!'})
    print('✅ Test notification sent!')
else:
    print('❌ DISCORD_WEBHOOK_URL not set')
"
```

## 🎯 **Training Commands**

### **Neural CF Baseline**
```bash
# Option 1: Your original script (auto-detects NCF)
python ../auto_train_with_notification.py

# Option 2: Enhanced script with explicit model selection
python auto_train_ss4rec.py --model ncf

# Expected Discord notification:
# 🎯 Best RMSE: 0.87 (✅ Good baseline)
```

### **SS4Rec SOTA** (After implementation)
```bash
python auto_train_ss4rec.py --model ss4rec

# Expected Discord notification:
# 🎯 Best RMSE: 0.64 (🏆 SOTA achieved!)
```

## 📊 **Discord Notification Examples**

### **Training Start**
```
🚀 Training Started - NCF
📊 Model: NCF (Baseline)
⏰ Started: 14:30 UTC
🖥️ System: Instance: runpod-xyz | GPU: NVIDIA A6000
🎯 Target: Validation RMSE < 0.90
⏳ Training in progress...
```

### **Training Complete (Success)**
```
🎉 MovieLens RecSys - NCF - Training Complete!
⏱️ Duration: 2.3 hours
🎯 Best RMSE: 0.87 (✅ Good baseline)
🖥️ System: Instance: runpod-xyz | GPU: NVIDIA A6000
📊 Check your W&B dashboard for detailed metrics
✅ Instance can now be safely terminated.
```

### **Training Failed**
```
❌ MovieLens RecSys - NCF - Training Failed
⏱️ Duration: 0.5 hours
❌ Exit Code: 1
📝 Check log file for details: training_ncf_20250118_143022.log
🔧 Common fixes:
• Check GPU memory: nvidia-smi
• Verify data files: ls data/processed/
```

## 🛠️ **Troubleshooting**

### **Script Not Found Error**
```bash
# If you get: "Training script not found: runpod_training_wandb.py"
ls -la runpod_training_wandb.py  # Should exist now
```

### **Discord Webhook Not Set**
```bash
# If you get: "No Discord webhook URL found"
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/YOUR_WEBHOOK"
echo $DISCORD_WEBHOOK_URL  # Verify it's set
```

### **W&B Issues**
```bash
# If W&B fails to initialize
wandb login  # Re-authenticate
# Or disable W&B:
python auto_train_ss4rec.py --model ncf --no-wandb
```

## 🔄 **Migration from Old Setup**

### **What Changed**
- ❌ `train_a6000.py` → ✅ `runpod_training_wandb.py`
- ❌ Multiple training scripts → ✅ Single unified script
- ❌ VAE focus → ✅ NCF vs SS4Rec comparison
- ❌ Broken Discord integration → ✅ Enhanced notifications

### **Your Workflow**
```bash
# 1. Set webhook (one time)
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/YOUR_WEBHOOK"

# 2. Start training (your exact same command works!)
python ../auto_train_with_notification.py

# 3. Close terminal, get Discord notification when done ✅
```

## 🎯 **Success Validation**

### **✅ Integration Working If:**
- Discord notification sent at training start
- Training logs appear in file
- W&B dashboard updates (if enabled)
- Discord notification sent at completion with RMSE results
- Model saved in `results/ncf_baseline/` or `results/ss4rec/`

### **Expected Results**
- **Neural CF**: 2-3 hours, RMSE ~0.85-0.90
- **SS4Rec**: 4-5 hours, RMSE ~0.60-0.70 (target)

---

**🔔 Your Discord auto-training setup is now fully compatible with the cleaned SS4Rec research focus!**