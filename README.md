# **FastAPI Project Setup & Deployment Guide**

This guide will help you set up and run the **FastAPI Transcriber** project on a Linux server.

---

## **1. Activating the Virtual Environment (`venv`)**

Before running the project, ensure the virtual environment is activated.

```bash
cd /home/abhilash/transcriber/transcriber  # Navigate to project directory
source venv/bin/activate  # Activate virtual environment
```

To confirm activation, check the Python version:

```bash
python --version
```

You should see Python running from `venv`, something like:

```
Python 3.x.x (/home/abhilash/transcriber/transcriber/venv/bin/python)
```

---

## **2. Installing Requirements**

Once the virtual environment is activated, install the required dependencies:

```bash
pip install -r requirements.txt
```

Ensure that all required packages are installed successfully.

---

## **3. Running the FastAPI Server with `nohup`**

To run the server persistently in the background, use `nohup`:

```bash
nohup /home/abhilash/transcriber/transcriber/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4 > /home/abhilash/transcriber/logs/transcriber-server.log 2>&1 &
```

### **Explanation:**

- `nohup` → Ensures the server keeps running after SSH disconnects.
- `/home/abhilash/transcriber/transcriber/venv/bin/uvicorn` → Runs Uvicorn from the virtual environment.
- `app.main:app` → Specifies the FastAPI app location (`app/main.py`).
- `--host 0.0.0.0` → Allows external access.
- `--port 8000` → Runs on port **8000**.
- `--workers 4` → Uses **4 worker processes** for better performance.
- `> /home/abhilash/transcriber/logs/transcriber-server.log 2>&1 &` → Redirects logs and runs the server in the background.

---

## **4. Checking If the Server is Running**

After running the command, check the **Uvicorn process**:

```bash
ps aux | grep uvicorn
```

If the server is running, you should see an output like:

```
abhilash   12345  0.5  2.3  123456 67890 ?     S    12:34   0:02 uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## **5. Viewing Logs**

To monitor logs in real-time:

```bash
tail -f /home/abhilash/transcriber/logs/transcriber-server.log
```

---

## **6. Stopping the Server**

Find the **Process ID (PID):**

```bash
ps aux | grep uvicorn
```

Then stop the server:

```bash
kill -9 <PID>
```

_(Replace `<PID>` with the actual process ID.)_

---

## **7. Auto-Start Server on Reboot**

To ensure the server starts automatically after a system reboot, add it to **crontab**:

1. Open crontab:
   ```bash
   crontab -e
   ```
2. Add this line at the bottom:
   ```bash
   @reboot nohup /home/abhilash/transcriber/transcriber/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4 > /home/abhilash/transcriber/logs/transcriber-server.log 2>&1 &
   ```
3. Save and exit.

---

✅ **Your FastAPI Transcriber server is now running persistently with `nohup`, and it will restart automatically on reboot!** 🚀
