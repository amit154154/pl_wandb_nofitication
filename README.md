# 📬 Telegram + Discord Notifications for Deep Learning Training  
*Because if your model’s improving, your wrist should know about it.*

---

## 🧠 What is this?

This is my first "vibe-coding" project. You know the kind—where you're not just solving a problem, you're creating a *feeling*.  

And that feeling is:  
> *“Ding! Epoch 7 finished — val_loss improved — here’s a graph on your wrist.”*

That's right. This project connects your **PyTorch Lightning model training** to your **Apple Watch** using Telegram (and optionally Discord), so you can:

- See real-time training updates  
- Get metrics and smooth graphs  
- Flex your overfitting in public places

---

## 📲 Why?

Because I was tired of opening a terminal, watching a slow loop, or checking W&B tabs.  
Now I just train my model and walk away — my **Apple Watch** tells me when something cool happens (like a drop in validation loss 😎).

---

## 💬 Features

### ✅ TelegramNotificationCallback
- Sends messages to **your Telegram bot**
- Text messages every epoch with metrics
- 3 cool-as-hell plots:
  1. **Train loss per epoch**
  2. **Train loss across current epoch** (smoothed like butter 🧈)
  3. **Validation loss per epoch**
- Works beautifully on Apple Watch (image previews and all)

### ✅ DiscordNotificationCallback *(optional)*
- Sends plain metric text + optional images to a **Discord webhook**
- No Apple Watch image previews (thanks Apple), but great for desktop/team alerts
