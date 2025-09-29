#!/usr/bin/env bash
cd artifact/Claims/cuckoo
./run.sh #(10 minutes)
cd .. 
cd mnist
./run.sh --attack FGSM # 1 hour for FGSM,
./run.sh --attack PGD # 1 hour for FGSM,
./run.sh --attack APGD-DLR # 1 hour for FGSM,
./run.sh --attack Square # 1 hour for FGSM,
./run.sh --attack SIT # 1 hour for FGSM,
cd ..
cd ember
./run.sh #(2.5 hours)
cd ..
cd cifar10
./run.sh #(2 hours)
