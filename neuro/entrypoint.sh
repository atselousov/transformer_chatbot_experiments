#!/usr/bin/env bash

sudo nohup docker daemon -H tcp://0.0.0.0:2375 -H unix:///var/run/docker.sock &
/usr/sbin/sshd -D