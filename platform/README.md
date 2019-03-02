Run the job on Neuromation platfrom with open http port
```bash
bash inference.sh
```
Register the bot
```bash
# TODO remove from history (Rauf 22.02.2019)
export MONGO_URI="mongodb+srv://admin:admin@cluster0-tgclb.mongodb.net/convai"

python convai_router_bot/system_monitor.py register-bot <bot_id> <telegram_bot_name>
```

Prepare router configuration
```bash
cat platform/bot_server_config.yml > convai_router_bot/config.yml
``` 
In `convai_router_bot/config.yml` replace job_id in `webhook` with your current job_id
Make sure `.git` folder is present it you are using remote 
deployment instead of `git clone` 

Run router app in tmux sessin
```bash
python convai_router_bot/application.py --port 8080
```

Get web hook
```bash
https://api.telegram.org/<telegram_bot_name>:<bot_id>/setWebhook?url=<job_url>
```

Check it
```bash
https://api.telegram.org/bot779933153:AAFi593HhCA_LlvOdFKU20yxJJ-Lq7X2FOw/getWebhookInfo
```

(Optional) Connect to db

```bash
mongo "mongodb+srv://cluster0-tgclb.mongodb.net/convai" --username admin
```

Run the agent in another tmux session
```bash
python wild.py -bi <bot_id> -rbu <job_url>
```
