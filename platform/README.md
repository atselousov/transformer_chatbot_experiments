Run job on the neuromation platform with open `--http 8080`

Register the bot
```bash
export MONGO_URI="mongodb+srv://admin:admin@cluster0-tgclb.mongodb.net/convai2"

python system_monitor.py register-bot <bot_id> <telegram_bot_name>
python system_monitor.py register-bot 779933153:AAFi593HhCA_LlvOdFKU20yxJJ-Lq7X2FOw kyryl_test_bot
```

Prepare router configuration
```bash
cat platfrom/bot_server_config.yml convai_router_bot/config.yml
``` 
In `config.yml` replace job_id in `webhook` with your current job_id
Make sure `.git` folder is present it you are using remote 
deployment instead of `git clone` 

Run router app in tmux sessin
```bash
python application.py --port 8080
```

Get web hook
```bash
https://api.telegram.org/kyryl_test_bot:779933153:AAFi593HhCA_LlvOdFKU20yxJJ-Lq7X2FOw/setWebhook?url=https://job-51f3e5c2-a48b-4c6f-906d-588537ca0997.jobs.platform.staging.neuromation.io
```

Check it
```bash
https://api.telegram.org/bot779933153:AAFi593HhCA_LlvOdFKU20yxJJ-Lq7X2FOw/getWebhookInfo
```

(Optional) Connect to db

```bash
mongo "mongodb+srv://cluster0-tgclb.mongodb.net/test" --username admin
```

Run the agent in another tmux session
```bash
python wild.py -bi <bot_id> -rbu <job_url>
python wild.py -bi bot779933153:AAFi593HhCA_LlvOdFKU20yxJJ-Lq7X2FOw -rbu python wild.py -bi bot779933153:AAFi593HhCA_LlvOdFKU20yxJJ-Lq7X2FOw -rbu https://job-02231c4c-23b7-4cdf-9cee-7ceaa0c135f0.jobs-staging.neu.ro

```
