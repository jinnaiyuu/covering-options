#/bin/bash

# task="Amidar-v0"
# task="MsPacman-v0"
# task="MontezumaRevenge-v0"
# task="BankHeist-v0"

# tasks="Amidar-v0 MsPacman-v0 BankHeist-v0"
tasks="Amidar-v0 MsPacman-v0 MontezumaRevenge-v0"

# TODO: Run Reversed direction too.

rev="--reverse"

for task in $tasks
do
    # python3 main.py --snsteps 2000 --snepisodes 50 --tasktype atari --task $task --ffuncnunit 256 --exp sample --noptions 0 --ffunction nnc --savecmp --sptrainingstep 100
    for i in 1 2 3 4
    do
	# python3 main.py --snsteps 1000 --snepisodes 100 --tasktype atari --task $task --ffuncnunit 256 --exp generate --noptions $i --ffunction nnc --savecmp --sptrainingstep 100
	# python3 main.py --snsteps 1000 --snepisodes 100 --tasktype atari --task $task --ffuncnunit 256 --exp train --noptions $i --ffunction nnc --savecmp --sptrainingstep 30
	python3 main.py --snsteps 1000 --snepisodes 20 --tasktype atari --task $task --ffuncnunit 256 --exp sample --noptions $i --ffunction nnc --savecmp --sptrainingstep 100 $rev
	python3 main.py --snsteps 1000 --snepisodes 100 --tasktype atari --task $task --ffuncnunit 256 --exp visterm --noptions $i --ffunction nnc --savecmp --sptrainingstep 30 $rev
    done
done

