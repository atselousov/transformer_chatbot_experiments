#!/usr/bin/env bash

python convai_router_bot/application.py --port 8080 &
#python wild.py -bi ${BOT_ID} -rbu ${JOB_URL} &
/usr/sbin/sshd -D &
$@
