#!/bin/bash

pbs delete_tasks
pbs add_tasks --tasks-file ../tasks/tasks.json --redundancy 1