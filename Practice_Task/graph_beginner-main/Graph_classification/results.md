# 图分类任务运行结果

(globalwork) E:\Aljx\graph_beginner-main\Graph_classification\code>python train_batch.py --dataset MUTAG --model GIN --pool avg --epochs 5 --batch_size 32
[Mini-batch] Device: cuda
Downloading https://www.chrsmrrs.com/graphkerneldatasets/MUTAG.zip
Processing...
Done!
Epoch   1 | TrainLoss 0.6751 | ValLoss 0.6604 | ValMetric 0.8333 | BestTest 0.7000
[Mini-batch] Best epoch 1, Test metric 0.7000, Total time 0.5s
{
  "best_val": 0.8333333333333334,
  "best_test": 0.7,
  "best_epoch": 1,
  "total_time": 0.4577765464782715
}

(globalwork) E:\Aljx\graph_beginner-main\Graph_classification\code>python run_experiments.py MUTAG 100
>>> ...\.conda\envs\globalwork\python.exe train_full.py --dataset MUTAG --model GCN --pool avg --epochs 100 --hidden 64 --layers 3 --lr 0.001
[Full-batch] Device: cuda
Epoch   1 | TrainLoss 0.6778 | ValLoss 0.6614 | ValMetric 0.6667 | BestTest 0.8000
Epoch  10 | TrainLoss 0.6535 | ValLoss 0.6545 | ValMetric 0.6667 | BestTest 0.8000
Epoch  20 | TrainLoss 0.6070 | ValLoss 0.6349 | ValMetric 0.6667 | BestTest 0.8000
Epoch  30 | TrainLoss 0.5851 | ValLoss 0.5763 | ValMetric 0.8889 | BestTest 0.8500
Epoch  40 | TrainLoss 0.5622 | ValLoss 0.4707 | ValMetric 0.8889 | BestTest 0.8500
Epoch  50 | TrainLoss 0.5278 | ValLoss 0.4248 | ValMetric 0.8889 | BestTest 0.8500
Epoch  60 | TrainLoss 0.5515 | ValLoss 0.4092 | ValMetric 0.8889 | BestTest 0.8500
Epoch  70 | TrainLoss 0.5313 | ValLoss 0.3959 | ValMetric 0.8889 | BestTest 0.8500
Epoch  80 | TrainLoss 0.5676 | ValLoss 0.3881 | ValMetric 0.8889 | BestTest 0.8500
Epoch  90 | TrainLoss 0.4960 | ValLoss 0.3873 | ValMetric 0.8889 | BestTest 0.8500
Epoch 100 | TrainLoss 0.5273 | ValLoss 0.3894 | ValMetric 0.8889 | BestTest 0.8500
[Full-batch] Best epoch 30, Test metric 0.8500, Total time 1.6s
{
  "best_val": 0.8888888888888888,
  "best_test": 0.85,
  "best_epoch": 30,
  "total_time": 1.6390924453735352
}
>>> ...\.conda\envs\globalwork\python.exe train_batch.py --dataset MUTAG --model GCN --pool avg --epochs 100 --hidden 64 --layers 3 --lr 0.001 --batch_size 32
[Mini-batch] Device: cuda
Epoch   1 | TrainLoss 0.6927 | ValLoss 0.7105 | ValMetric 0.1667 | BestTest 0.2000
Epoch  10 | TrainLoss 0.5349 | ValLoss 0.4128 | ValMetric 0.8333 | BestTest 0.8000
Epoch  20 | TrainLoss 0.5541 | ValLoss 0.3663 | ValMetric 0.7778 | BestTest 0.8000
Epoch  30 | TrainLoss 0.5304 | ValLoss 0.3611 | ValMetric 0.8333 | BestTest 0.8000
Epoch  40 | TrainLoss 0.5246 | ValLoss 0.3533 | ValMetric 0.7778 | BestTest 0.8000
Epoch  50 | TrainLoss 0.5475 | ValLoss 0.3539 | ValMetric 0.8333 | BestTest 0.8000
Epoch  60 | TrainLoss 0.5592 | ValLoss 0.3504 | ValMetric 0.7778 | BestTest 0.8000
Epoch  70 | TrainLoss 0.5088 | ValLoss 0.3494 | ValMetric 0.8333 | BestTest 0.8000
Epoch  80 | TrainLoss 0.5163 | ValLoss 0.3442 | ValMetric 0.8333 | BestTest 0.8000
Epoch  90 | TrainLoss 0.5313 | ValLoss 0.3450 | ValMetric 0.8333 | BestTest 0.8000
Epoch 100 | TrainLoss 0.5251 | ValLoss 0.3417 | ValMetric 0.8333 | BestTest 0.8000
[Mini-batch] Best epoch 2, Test metric 0.8000, Total time 3.7s
{
  "best_val": 0.8333333333333334,
  "best_test": 0.8,
  "best_epoch": 2,
  "total_time": 3.733776330947876
}
>>> ...\.conda\envs\globalwork\python.exe train_full.py --dataset MUTAG --model GCN --pool max --epochs 100 --hidden 64 --layers 3 --lr 0.001
[Full-batch] Device: cuda
Epoch   1 | TrainLoss 0.9052 | ValLoss 0.7042 | ValMetric 0.3333 | BestTest 0.2000
Epoch  10 | TrainLoss 0.6240 | ValLoss 0.6856 | ValMetric 0.7222 | BestTest 0.8000
Epoch  20 | TrainLoss 0.6322 | ValLoss 0.6670 | ValMetric 0.6667 | BestTest 0.8000
Epoch  30 | TrainLoss 0.6133 | ValLoss 0.6438 | ValMetric 0.6667 | BestTest 0.8000
Epoch  40 | TrainLoss 0.5499 | ValLoss 0.6225 | ValMetric 0.7222 | BestTest 0.8000
Epoch  50 | TrainLoss 0.5617 | ValLoss 0.5873 | ValMetric 0.7778 | BestTest 0.8000
Epoch  60 | TrainLoss 0.5357 | ValLoss 0.5674 | ValMetric 0.7778 | BestTest 0.8000
Epoch  70 | TrainLoss 0.5570 | ValLoss 0.5572 | ValMetric 0.7778 | BestTest 0.8000
Epoch  80 | TrainLoss 0.4826 | ValLoss 0.5558 | ValMetric 0.7222 | BestTest 0.8000
Epoch  90 | TrainLoss 0.5167 | ValLoss 0.5572 | ValMetric 0.7778 | BestTest 0.8000
Epoch 100 | TrainLoss 0.5445 | ValLoss 0.5525 | ValMetric 0.7778 | BestTest 0.8000
[Full-batch] Best epoch 50, Test metric 0.8000, Total time 1.6s
{
  "best_val": 0.7777777777777778,
  "best_test": 0.8,
  "best_epoch": 50,
  "total_time": 1.576192855834961
}
>>> ...\.conda\envs\globalwork\python.exe train_batch.py --dataset MUTAG --model GCN --pool max --epochs 100 --hidden 64 --layers 3 --lr 0.001 --batch_size 32
[Mini-batch] Device: cuda
Epoch   1 | TrainLoss 1.1980 | ValLoss 0.6976 | ValMetric 0.4444 | BestTest 0.5500
Epoch  10 | TrainLoss 0.5785 | ValLoss 0.6939 | ValMetric 0.4444 | BestTest 0.5000
Epoch  20 | TrainLoss 0.5191 | ValLoss 0.6329 | ValMetric 0.6111 | BestTest 0.5500
Epoch  30 | TrainLoss 0.5211 | ValLoss 0.6215 | ValMetric 0.6667 | BestTest 0.5500
Epoch  40 | TrainLoss 0.4305 | ValLoss 0.6493 | ValMetric 0.5556 | BestTest 0.5500
Epoch  50 | TrainLoss 0.4738 | ValLoss 0.6268 | ValMetric 0.6111 | BestTest 0.5500
Epoch  60 | TrainLoss 0.4204 | ValLoss 0.6321 | ValMetric 0.6111 | BestTest 0.5500
Epoch  70 | TrainLoss 0.4569 | ValLoss 0.6400 | ValMetric 0.6111 | BestTest 0.5500
Epoch  80 | TrainLoss 0.4469 | ValLoss 0.6345 | ValMetric 0.6111 | BestTest 0.5500
Epoch  90 | TrainLoss 0.4643 | ValLoss 0.6270 | ValMetric 0.6111 | BestTest 0.5500
Epoch 100 | TrainLoss 0.4510 | ValLoss 0.6410 | ValMetric 0.5556 | BestTest 0.5500
[Mini-batch] Best epoch 16, Test metric 0.5500, Total time 3.8s
{
  "best_val": 0.6666666666666666,
  "best_test": 0.55,
  "best_epoch": 16,
  "total_time": 3.7815299034118652
}
>>> ...\.conda\envs\globalwork\python.exe train_full.py --dataset MUTAG --model GCN --pool min --epochs 100 --hidden 64 --layers 3 --lr 0.001
[Full-batch] Device: cuda
Epoch   1 | TrainLoss 0.6898 | ValLoss 0.6880 | ValMetric 0.6667 | BestTest 0.6000
Epoch  10 | TrainLoss 0.6819 | ValLoss 0.6788 | ValMetric 0.6667 | BestTest 0.6000
Epoch  20 | TrainLoss 0.6685 | ValLoss 0.6686 | ValMetric 0.6667 | BestTest 0.6000
Epoch  30 | TrainLoss 0.6603 | ValLoss 0.6596 | ValMetric 0.6667 | BestTest 0.6000
Epoch  40 | TrainLoss 0.6551 | ValLoss 0.6529 | ValMetric 0.6667 | BestTest 0.6000
Epoch  50 | TrainLoss 0.6462 | ValLoss 0.6481 | ValMetric 0.6667 | BestTest 0.6000
Epoch  60 | TrainLoss 0.6424 | ValLoss 0.6463 | ValMetric 0.6667 | BestTest 0.6000
Epoch  70 | TrainLoss 0.6430 | ValLoss 0.6435 | ValMetric 0.6667 | BestTest 0.6000
Epoch  80 | TrainLoss 0.6414 | ValLoss 0.6410 | ValMetric 0.6667 | BestTest 0.6000
Epoch  90 | TrainLoss 0.6294 | ValLoss 0.6393 | ValMetric 0.6667 | BestTest 0.6000
Epoch 100 | TrainLoss 0.6315 | ValLoss 0.6378 | ValMetric 0.6667 | BestTest 0.6000
[Full-batch] Best epoch 1, Test metric 0.6000, Total time 1.6s
{
  "best_val": 0.6666666666666666,
  "best_test": 0.6,
  "best_epoch": 1,
  "total_time": 1.5657196044921875
}
>>> ...\.conda\envs\globalwork\python.exe train_batch.py --dataset MUTAG --model GCN --pool min --epochs 100 --hidden 64 --layers 3 --lr 0.001 --batch_size 32
[Mini-batch] Device: cuda
Epoch   1 | TrainLoss 0.6969 | ValLoss 0.7034 | ValMetric 0.2778 | BestTest 0.3500
Epoch  10 | TrainLoss 0.6635 | ValLoss 0.6489 | ValMetric 0.7222 | BestTest 0.6500
Epoch  20 | TrainLoss 0.6456 | ValLoss 0.6141 | ValMetric 0.7222 | BestTest 0.6500
Epoch  30 | TrainLoss 0.6429 | ValLoss 0.6032 | ValMetric 0.7222 | BestTest 0.6500
Epoch  40 | TrainLoss 0.6385 | ValLoss 0.6015 | ValMetric 0.7222 | BestTest 0.6500
Epoch  50 | TrainLoss 0.6444 | ValLoss 0.6012 | ValMetric 0.7222 | BestTest 0.6500
Epoch  60 | TrainLoss 0.6360 | ValLoss 0.6013 | ValMetric 0.7222 | BestTest 0.6500
Epoch  70 | TrainLoss 0.6415 | ValLoss 0.6011 | ValMetric 0.7222 | BestTest 0.6500
Epoch  80 | TrainLoss 0.6415 | ValLoss 0.6014 | ValMetric 0.7222 | BestTest 0.6500
Epoch  90 | TrainLoss 0.6347 | ValLoss 0.6013 | ValMetric 0.7222 | BestTest 0.6500
Epoch 100 | TrainLoss 0.6390 | ValLoss 0.6007 | ValMetric 0.7222 | BestTest 0.6500
[Mini-batch] Best epoch 4, Test metric 0.6500, Total time 3.8s
{
  "best_val": 0.7222222222222222,
  "best_test": 0.65,
  "best_epoch": 4,
  "total_time": 3.768519163131714
}
>>> ...\.conda\envs\globalwork\python.exe train_full.py --dataset MUTAG --model GAT --pool avg --epochs 100 --hidden 64 --layers 3 --lr 0.001
[Full-batch] Device: cuda
Epoch   1 | TrainLoss 0.8006 | ValLoss 0.7522 | ValMetric 0.2778 | BestTest 0.3000
Epoch  10 | TrainLoss 0.6849 | ValLoss 0.7164 | ValMetric 0.2778 | BestTest 0.3000
Epoch  20 | TrainLoss 0.6670 | ValLoss 0.6658 | ValMetric 0.7222 | BestTest 0.7000
Epoch  30 | TrainLoss 0.6584 | ValLoss 0.6450 | ValMetric 0.7222 | BestTest 0.7000
Epoch  40 | TrainLoss 0.6560 | ValLoss 0.6384 | ValMetric 0.7222 | BestTest 0.7000
Epoch  50 | TrainLoss 0.6342 | ValLoss 0.6266 | ValMetric 0.7222 | BestTest 0.7000
Epoch  60 | TrainLoss 0.6493 | ValLoss 0.6157 | ValMetric 0.7222 | BestTest 0.7000
Epoch  70 | TrainLoss 0.6374 | ValLoss 0.6072 | ValMetric 0.7222 | BestTest 0.7000
Epoch  80 | TrainLoss 0.6312 | ValLoss 0.5995 | ValMetric 0.7222 | BestTest 0.7000
Epoch  90 | TrainLoss 0.6097 | ValLoss 0.5944 | ValMetric 0.7222 | BestTest 0.7000
Epoch 100 | TrainLoss 0.6258 | ValLoss 0.5897 | ValMetric 0.7222 | BestTest 0.7000
[Full-batch] Best epoch 15, Test metric 0.7000, Total time 1.8s
{
  "best_val": 0.7222222222222222,
  "best_test": 0.7,
  "best_epoch": 15,
  "total_time": 1.8129973411560059
}
>>> ...\.conda\envs\globalwork\python.exe train_batch.py --dataset MUTAG --model GAT --pool avg --epochs 100 --hidden 64 --layers 3 --lr 0.001 --batch_size 32
[Mini-batch] Device: cuda
Epoch   1 | TrainLoss 0.7288 | ValLoss 0.7196 | ValMetric 0.3333 | BestTest 0.4000
Epoch  10 | TrainLoss 0.6443 | ValLoss 0.6523 | ValMetric 0.6667 | BestTest 0.6000
Epoch  20 | TrainLoss 0.6047 | ValLoss 0.6202 | ValMetric 0.6667 | BestTest 0.6000
Epoch  30 | TrainLoss 0.6037 | ValLoss 0.5809 | ValMetric 0.6667 | BestTest 0.6000
Epoch  40 | TrainLoss 0.5825 | ValLoss 0.5339 | ValMetric 0.7222 | BestTest 0.6500
Epoch  50 | TrainLoss 0.5635 | ValLoss 0.5048 | ValMetric 0.7778 | BestTest 0.6500
Epoch  60 | TrainLoss 0.5290 | ValLoss 0.5063 | ValMetric 0.7222 | BestTest 0.6500
Epoch  70 | TrainLoss 0.5303 | ValLoss 0.5069 | ValMetric 0.7222 | BestTest 0.6500
Epoch  80 | TrainLoss 0.5214 | ValLoss 0.5127 | ValMetric 0.6667 | BestTest 0.6500
Epoch  90 | TrainLoss 0.5667 | ValLoss 0.5011 | ValMetric 0.7778 | BestTest 0.6500
Epoch 100 | TrainLoss 0.5206 | ValLoss 0.4951 | ValMetric 0.7778 | BestTest 0.6500
[Mini-batch] Best epoch 44, Test metric 0.6500, Total time 5.0s
{
  "best_val": 0.7777777777777778,
  "best_test": 0.65,
  "best_epoch": 44,
  "total_time": 4.986523866653442
}
>>> ...\.conda\envs\globalwork\python.exe train_full.py --dataset MUTAG --model GAT --pool max --epochs 100 --hidden 64 --layers 3 --lr 0.001
[Full-batch] Device: cuda
Epoch   1 | TrainLoss 0.7425 | ValLoss 0.6669 | ValMetric 0.7778 | BestTest 0.6500
Epoch  10 | TrainLoss 0.6530 | ValLoss 0.6707 | ValMetric 0.7778 | BestTest 0.6500
Epoch  20 | TrainLoss 0.7260 | ValLoss 0.6705 | ValMetric 0.7778 | BestTest 0.6500
Epoch  30 | TrainLoss 0.6746 | ValLoss 0.6685 | ValMetric 0.7778 | BestTest 0.6500
Epoch  40 | TrainLoss 0.6516 | ValLoss 0.6673 | ValMetric 0.7778 | BestTest 0.6500
Epoch  50 | TrainLoss 0.6441 | ValLoss 0.6661 | ValMetric 0.7778 | BestTest 0.6500
Epoch  60 | TrainLoss 0.6428 | ValLoss 0.6689 | ValMetric 0.7778 | BestTest 0.6500
Epoch  70 | TrainLoss 0.6358 | ValLoss 0.6711 | ValMetric 0.7778 | BestTest 0.6500
Epoch  80 | TrainLoss 0.6421 | ValLoss 0.6690 | ValMetric 0.7778 | BestTest 0.6500
Epoch  90 | TrainLoss 0.6631 | ValLoss 0.6689 | ValMetric 0.7778 | BestTest 0.6500
Epoch 100 | TrainLoss 0.6473 | ValLoss 0.6708 | ValMetric 0.7778 | BestTest 0.6500
[Full-batch] Best epoch 1, Test metric 0.6500, Total time 1.9s
{
  "best_val": 0.7777777777777778,
  "best_test": 0.65,
  "best_epoch": 1,
  "total_time": 1.902839183807373
}
>>> ...\.conda\envs\globalwork\python.exe train_batch.py --dataset MUTAG --model GAT --pool max --epochs 100 --hidden 64 --layers 3 --lr 0.001 --batch_size 32
[Mini-batch] Device: cuda
Epoch   1 | TrainLoss 0.7146 | ValLoss 0.6766 | ValMetric 0.6667 | BestTest 0.6500
Epoch  10 | TrainLoss 0.6317 | ValLoss 0.6674 | ValMetric 0.6667 | BestTest 0.6500
Epoch  20 | TrainLoss 0.5950 | ValLoss 0.6719 | ValMetric 0.6667 | BestTest 0.6500
Epoch  30 | TrainLoss 0.6281 | ValLoss 0.6647 | ValMetric 0.6667 | BestTest 0.6500
Epoch  40 | TrainLoss 0.6097 | ValLoss 0.6715 | ValMetric 0.7778 | BestTest 0.7000
Epoch  50 | TrainLoss 0.5772 | ValLoss 0.6763 | ValMetric 0.8889 | BestTest 0.7500
Epoch  60 | TrainLoss 0.6211 | ValLoss 0.6721 | ValMetric 0.8889 | BestTest 0.7500
Epoch  70 | TrainLoss 0.5772 | ValLoss 0.6748 | ValMetric 0.8889 | BestTest 0.7500
Epoch  80 | TrainLoss 0.6012 | ValLoss 0.6695 | ValMetric 0.8889 | BestTest 0.7500
Epoch  90 | TrainLoss 0.5926 | ValLoss 0.6758 | ValMetric 0.6667 | BestTest 0.7500
Epoch 100 | TrainLoss 0.5385 | ValLoss 0.6739 | ValMetric 0.8333 | BestTest 0.7500
[Mini-batch] Best epoch 50, Test metric 0.7500, Total time 4.8s
{
  "best_val": 0.8888888888888888,
  "best_test": 0.75,
  "best_epoch": 50,
  "total_time": 4.772607326507568
}
>>> ...\.conda\envs\globalwork\python.exe train_full.py --dataset MUTAG --model GAT --pool min --epochs 100 --hidden 64 --layers 3 --lr 0.001
[Full-batch] Device: cuda
>>> C:\Users\XiaoHui\.conda\envs\globalwork\python.exe train_batch.py --dataset MUTAG --model GAT --pool min --epochs 100 --hidden 64 --layers 3 --lr 0.001 --batch_size 32
[Mini-batch] Device: cuda
Epoch   1 | TrainLoss 0.6929 | ValLoss 0.6930 | ValMetric 0.4444 | BestTest 0.3500
Epoch  10 | TrainLoss 0.6428 | ValLoss 0.6879 | ValMetric 0.5556 | BestTest 0.6500
Epoch  20 | TrainLoss 0.6304 | ValLoss 0.7100 | ValMetric 0.5556 | BestTest 0.6500
Epoch  30 | TrainLoss 0.6344 | ValLoss 0.7176 | ValMetric 0.5556 | BestTest 0.6500
Epoch  40 | TrainLoss 0.6238 | ValLoss 0.7185 | ValMetric 0.5556 | BestTest 0.6500
Epoch  50 | TrainLoss 0.6291 | ValLoss 0.7202 | ValMetric 0.5556 | BestTest 0.6500
Epoch  60 | TrainLoss 0.6364 | ValLoss 0.7184 | ValMetric 0.5556 | BestTest 0.6500
Epoch  70 | TrainLoss 0.6267 | ValLoss 0.7193 | ValMetric 0.5556 | BestTest 0.6500
Epoch  80 | TrainLoss 0.6264 | ValLoss 0.7199 | ValMetric 0.5556 | BestTest 0.6500
Epoch  90 | TrainLoss 0.6271 | ValLoss 0.7193 | ValMetric 0.5556 | BestTest 0.6500
Epoch 100 | TrainLoss 0.6290 | ValLoss 0.7198 | ValMetric 0.5556 | BestTest 0.6500
[Mini-batch] Best epoch 2, Test metric 0.6500, Total time 4.8s
{
  "best_val": 0.5555555555555556,
  "best_test": 0.65,
  "best_epoch": 2,
  "total_time": 4.77297306060791
}
>>> ...\.conda\envs\globalwork\python.exe train_full.py --dataset MUTAG --model SAGE --pool avg --epochs 100 --hidden 64 --layers 3 --lr 0.001
[Full-batch] Device: cuda
Epoch   1 | TrainLoss 0.6569 | ValLoss 0.6296 | ValMetric 0.7778 | BestTest 0.5000
Epoch  10 | TrainLoss 0.6265 | ValLoss 0.5920 | ValMetric 0.7778 | BestTest 0.5000
Epoch  20 | TrainLoss 0.5933 | ValLoss 0.5350 | ValMetric 0.7778 | BestTest 0.5000
Epoch  30 | TrainLoss 0.5341 | ValLoss 0.5226 | ValMetric 0.7778 | BestTest 0.5000
Epoch  40 | TrainLoss 0.5221 | ValLoss 0.5560 | ValMetric 0.7778 | BestTest 0.5000
Epoch  50 | TrainLoss 0.5224 | ValLoss 0.5654 | ValMetric 0.7778 | BestTest 0.5000
Epoch  60 | TrainLoss 0.5156 | ValLoss 0.5537 | ValMetric 0.7778 | BestTest 0.5000
Epoch  70 | TrainLoss 0.4931 | ValLoss 0.5357 | ValMetric 0.7778 | BestTest 0.5000
Epoch  80 | TrainLoss 0.5091 | ValLoss 0.5387 | ValMetric 0.7778 | BestTest 0.5000
Epoch  90 | TrainLoss 0.4808 | ValLoss 0.5440 | ValMetric 0.7778 | BestTest 0.5000
Epoch 100 | TrainLoss 0.5370 | ValLoss 0.5443 | ValMetric 0.7778 | BestTest 0.5000
[Full-batch] Best epoch 1, Test metric 0.5000, Total time 1.4s
{
  "best_val": 0.7777777777777778,
  "best_test": 0.5,
  "best_epoch": 1,
  "total_time": 1.4232089519500732
}
>>> ...\.conda\envs\globalwork\python.exe train_batch.py --dataset MUTAG --model SAGE --pool avg --epochs 100 --hidden 64 --layers 3 --lr 0.001 --batch_size 32
[Mini-batch] Device: cuda
Epoch   1 | TrainLoss 0.6850 | ValLoss 0.6970 | ValMetric 0.2222 | BestTest 0.3500
Epoch  10 | TrainLoss 0.5442 | ValLoss 0.4231 | ValMetric 0.7778 | BestTest 0.6500
Epoch  20 | TrainLoss 0.5137 | ValLoss 0.3806 | ValMetric 0.8333 | BestTest 0.8000
Epoch  30 | TrainLoss 0.5136 | ValLoss 0.3899 | ValMetric 0.8333 | BestTest 0.8000
Epoch  40 | TrainLoss 0.5155 | ValLoss 0.3880 | ValMetric 0.8333 | BestTest 0.8000
Epoch  50 | TrainLoss 0.5142 | ValLoss 0.3895 | ValMetric 0.8333 | BestTest 0.8000
Epoch  60 | TrainLoss 0.5298 | ValLoss 0.3940 | ValMetric 0.8333 | BestTest 0.8000
Epoch  70 | TrainLoss 0.5055 | ValLoss 0.3992 | ValMetric 0.8333 | BestTest 0.8000
Epoch  80 | TrainLoss 0.4893 | ValLoss 0.4016 | ValMetric 0.8333 | BestTest 0.8000
Epoch  90 | TrainLoss 0.4984 | ValLoss 0.4048 | ValMetric 0.8333 | BestTest 0.8000
Epoch 100 | TrainLoss 0.4857 | ValLoss 0.4068 | ValMetric 0.8333 | BestTest 0.8000
[Mini-batch] Best epoch 13, Test metric 0.8000, Total time 3.2s
{
  "best_val": 0.9444444444444444,
  "best_test": 0.8,
  "best_epoch": 13,
  "total_time": 3.209663152694702
}
>>> ...\.conda\envs\globalwork\python.exe train_full.py --dataset MUTAG --model SAGE --pool max --epochs 100 --hidden 64 --layers 3 --lr 0.001
[Full-batch] Device: cuda
Epoch   1 | TrainLoss 0.6798 | ValLoss 0.6776 | ValMetric 0.7778 | BestTest 0.5000
Epoch  10 | TrainLoss 0.6312 | ValLoss 0.6687 | ValMetric 0.7778 | BestTest 0.5000
Epoch  20 | TrainLoss 0.6659 | ValLoss 0.6440 | ValMetric 0.7778 | BestTest 0.5000
Epoch  30 | TrainLoss 0.6261 | ValLoss 0.6197 | ValMetric 0.7778 | BestTest 0.5000
Epoch  40 | TrainLoss 0.6253 | ValLoss 0.5921 | ValMetric 0.7778 | BestTest 0.5000
Epoch  50 | TrainLoss 0.5716 | ValLoss 0.5492 | ValMetric 0.7778 | BestTest 0.5000
Epoch  60 | TrainLoss 0.5865 | ValLoss 0.5234 | ValMetric 0.8333 | BestTest 0.5000
Epoch  70 | TrainLoss 0.5904 | ValLoss 0.5053 | ValMetric 0.8889 | BestTest 0.5500
Epoch  80 | TrainLoss 0.5772 | ValLoss 0.4854 | ValMetric 1.0000 | BestTest 0.6000
Epoch  90 | TrainLoss 0.5564 | ValLoss 0.4760 | ValMetric 1.0000 | BestTest 0.6000
Epoch 100 | TrainLoss 0.5507 | ValLoss 0.4655 | ValMetric 1.0000 | BestTest 0.6000
[Full-batch] Best epoch 77, Test metric 0.6000, Total time 1.4s
{
  "best_val": 1.0,
  "best_test": 0.6,
  "best_epoch": 77,
  "total_time": 1.3706603050231934
}
>>> ...\.conda\envs\globalwork\python.exe train_batch.py --dataset MUTAG --model SAGE --pool max --epochs 100 --hidden 64 --layers 3 --lr 0.001 --batch_size 32
[Mini-batch] Device: cuda
Epoch   1 | TrainLoss 0.7824 | ValLoss 0.6771 | ValMetric 0.6667 | BestTest 0.6500
Epoch  10 | TrainLoss 0.6180 | ValLoss 0.6531 | ValMetric 0.6667 | BestTest 0.6500
Epoch  20 | TrainLoss 0.5402 | ValLoss 0.6588 | ValMetric 0.6667 | BestTest 0.6500
Epoch  30 | TrainLoss 0.5586 | ValLoss 0.6635 | ValMetric 0.6111 | BestTest 0.6500
Epoch  40 | TrainLoss 0.4934 | ValLoss 0.6351 | ValMetric 0.6111 | BestTest 0.6500
Epoch  50 | TrainLoss 0.5001 | ValLoss 0.6409 | ValMetric 0.6111 | BestTest 0.6500
Epoch  60 | TrainLoss 0.4691 | ValLoss 0.6383 | ValMetric 0.6111 | BestTest 0.6500
Epoch  70 | TrainLoss 0.4558 | ValLoss 0.6402 | ValMetric 0.6111 | BestTest 0.6500
Epoch  80 | TrainLoss 0.4965 | ValLoss 0.6359 | ValMetric 0.6111 | BestTest 0.6500
Epoch  90 | TrainLoss 0.5373 | ValLoss 0.6381 | ValMetric 0.6111 | BestTest 0.6500
Epoch 100 | TrainLoss 0.4634 | ValLoss 0.6406 | ValMetric 0.6111 | BestTest 0.6500
[Mini-batch] Best epoch 1, Test metric 0.6500, Total time 3.2s
{
  "best_val": 0.6666666666666666,
  "best_test": 0.65,
  "best_epoch": 1,
  "total_time": 3.24053955078125
}
>>> ...\.conda\envs\globalwork\python.exe train_full.py --dataset MUTAG --model SAGE --pool min --epochs 100 --hidden 64 --layers 3 --lr 0.001
[Full-batch] Device: cuda
Epoch   1 | TrainLoss 0.7166 | ValLoss 0.7016 | ValMetric 0.4444 | BestTest 0.2500
Epoch  10 | TrainLoss 0.7028 | ValLoss 0.6985 | ValMetric 0.4444 | BestTest 0.2500
Epoch  20 | TrainLoss 0.6904 | ValLoss 0.6950 | ValMetric 0.3333 | BestTest 0.2500
Epoch  30 | TrainLoss 0.6746 | ValLoss 0.6905 | ValMetric 0.5556 | BestTest 0.7500
Epoch  40 | TrainLoss 0.6654 | ValLoss 0.6881 | ValMetric 0.5556 | BestTest 0.7500
Epoch  50 | TrainLoss 0.6596 | ValLoss 0.6874 | ValMetric 0.5556 | BestTest 0.7500
Epoch  60 | TrainLoss 0.6544 | ValLoss 0.6895 | ValMetric 0.5556 | BestTest 0.7500
Epoch  70 | TrainLoss 0.6482 | ValLoss 0.6908 | ValMetric 0.5556 | BestTest 0.7500
Epoch  80 | TrainLoss 0.6500 | ValLoss 0.6922 | ValMetric 0.5556 | BestTest 0.7500
Epoch  90 | TrainLoss 0.6445 | ValLoss 0.6939 | ValMetric 0.5556 | BestTest 0.7500
Epoch 100 | TrainLoss 0.6423 | ValLoss 0.6959 | ValMetric 0.5556 | BestTest 0.7500
[Full-batch] Best epoch 22, Test metric 0.7500, Total time 1.4s
{
  "best_val": 0.5555555555555556,
  "best_test": 0.75,
  "best_epoch": 22,
  "total_time": 1.3795714378356934
}
>>> ...\.conda\envs\globalwork\python.exe train_batch.py --dataset MUTAG --model SAGE --pool min --epochs 100 --hidden 64 --layers 3 --lr 0.001 --batch_size 32
[Mini-batch] Device: cuda
Epoch   1 | TrainLoss 0.7049 | ValLoss 0.7046 | ValMetric 0.2778 | BestTest 0.2500
Epoch  10 | TrainLoss 0.6703 | ValLoss 0.6462 | ValMetric 0.7222 | BestTest 0.7500
Epoch  20 | TrainLoss 0.6512 | ValLoss 0.6156 | ValMetric 0.7222 | BestTest 0.7500
Epoch  30 | TrainLoss 0.6515 | ValLoss 0.6076 | ValMetric 0.7222 | BestTest 0.7500
Epoch  40 | TrainLoss 0.6528 | ValLoss 0.6052 | ValMetric 0.7222 | BestTest 0.7500
Epoch  50 | TrainLoss 0.6532 | ValLoss 0.6053 | ValMetric 0.7222 | BestTest 0.7500
Epoch  60 | TrainLoss 0.6506 | ValLoss 0.6060 | ValMetric 0.7222 | BestTest 0.7500
Epoch  70 | TrainLoss 0.6497 | ValLoss 0.6051 | ValMetric 0.7222 | BestTest 0.7500
Epoch  80 | TrainLoss 0.6490 | ValLoss 0.6051 | ValMetric 0.7222 | BestTest 0.7500
Epoch  90 | TrainLoss 0.6555 | ValLoss 0.6052 | ValMetric 0.7222 | BestTest 0.7500
Epoch 100 | TrainLoss 0.6552 | ValLoss 0.6045 | ValMetric 0.7222 | BestTest 0.7500
[Mini-batch] Best epoch 3, Test metric 0.7500, Total time 3.1s
{
  "best_val": 0.7222222222222222,
  "best_test": 0.75,
  "best_epoch": 3,
  "total_time": 3.1168081760406494
}
>>> ...\.conda\envs\globalwork\python.exe train_full.py --dataset MUTAG --model GIN --pool avg --epochs 100 --hidden 64 --layers 3 --lr 0.001
[Full-batch] Device: cuda
Epoch   1 | TrainLoss 0.7009 | ValLoss 0.6926 | ValMetric 0.5556 | BestTest 0.4000
Epoch  10 | TrainLoss 0.5446 | ValLoss 0.7058 | ValMetric 0.4444 | BestTest 0.4000
Epoch  20 | TrainLoss 0.4317 | ValLoss 0.7334 | ValMetric 0.4444 | BestTest 0.4000
Epoch  30 | TrainLoss 0.3592 | ValLoss 0.8301 | ValMetric 0.5000 | BestTest 0.4000
Epoch  40 | TrainLoss 0.3221 | ValLoss 0.9225 | ValMetric 0.5000 | BestTest 0.4000
Epoch  50 | TrainLoss 0.2897 | ValLoss 1.1535 | ValMetric 0.5000 | BestTest 0.4000
Epoch  60 | TrainLoss 0.3051 | ValLoss 0.8333 | ValMetric 0.5000 | BestTest 0.4000
Epoch  70 | TrainLoss 0.2927 | ValLoss 0.7341 | ValMetric 0.6667 | BestTest 0.6000
Epoch  80 | TrainLoss 0.2585 | ValLoss 0.5426 | ValMetric 0.6667 | BestTest 0.8000
Epoch  90 | TrainLoss 0.3162 | ValLoss 0.5857 | ValMetric 0.6667 | BestTest 0.8000
Epoch 100 | TrainLoss 0.2499 | ValLoss 0.4558 | ValMetric 0.7778 | BestTest 0.8000
[Full-batch] Best epoch 75, Test metric 0.8000, Total time 1.3s
{
  "best_val": 0.7777777777777778,
  "best_test": 0.8,
  "best_epoch": 75,
  "total_time": 1.340888500213623
}
>>> ...\.conda\envs\globalwork\python.exe train_batch.py --dataset MUTAG --model GIN --pool avg --epochs 100 --hidden 64 --layers 3 --lr 0.001 --batch_size 32
[Mini-batch] Device: cuda
Epoch   1 | TrainLoss 0.6662 | ValLoss 0.6364 | ValMetric 0.7778 | BestTest 0.7000
Epoch  10 | TrainLoss 0.4094 | ValLoss 0.4390 | ValMetric 0.6667 | BestTest 0.7000
Epoch  20 | TrainLoss 0.3530 | ValLoss 0.4551 | ValMetric 0.7222 | BestTest 0.7000
Epoch  30 | TrainLoss 0.3386 | ValLoss 0.5443 | ValMetric 0.6667 | BestTest 0.7000
Epoch  40 | TrainLoss 0.2641 | ValLoss 0.4685 | ValMetric 0.6111 | BestTest 0.7000
Epoch  50 | TrainLoss 0.2776 | ValLoss 0.6284 | ValMetric 0.6667 | BestTest 0.7000
Epoch  60 | TrainLoss 0.2504 | ValLoss 0.5716 | ValMetric 0.6111 | BestTest 0.7000
Epoch  70 | TrainLoss 0.2284 | ValLoss 0.6608 | ValMetric 0.6667 | BestTest 0.7000
Epoch  80 | TrainLoss 0.2294 | ValLoss 0.5784 | ValMetric 0.6111 | BestTest 0.7000
Epoch  90 | TrainLoss 0.2893 | ValLoss 0.5847 | ValMetric 0.6667 | BestTest 0.7000
Epoch 100 | TrainLoss 0.2075 | ValLoss 0.6294 | ValMetric 0.7222 | BestTest 0.7000
[Mini-batch] Best epoch 1, Test metric 0.7000, Total time 3.2s
{
  "best_val": 0.7777777777777778,
  "best_test": 0.7,
  "best_epoch": 1,
  "total_time": 3.1659717559814453
}
>>> ...\.conda\envs\globalwork\python.exe train_full.py --dataset MUTAG --model GIN --pool max --epochs 100 --hidden 64 --layers 3 --lr 0.001
[Full-batch] Device: cuda
Epoch   1 | TrainLoss 0.7259 | ValLoss 0.6907 | ValMetric 0.6667 | BestTest 0.6500
Epoch  10 | TrainLoss 0.4565 | ValLoss 0.6405 | ValMetric 0.6667 | BestTest 0.6500
Epoch  20 | TrainLoss 0.4244 | ValLoss 0.5302 | ValMetric 0.6667 | BestTest 0.6500
Epoch  30 | TrainLoss 0.4387 | ValLoss 0.4491 | ValMetric 0.6111 | BestTest 0.6500
Epoch  40 | TrainLoss 0.3997 | ValLoss 0.3962 | ValMetric 0.8333 | BestTest 0.7500
Epoch  50 | TrainLoss 0.3863 | ValLoss 0.3917 | ValMetric 0.8333 | BestTest 0.7500
Epoch  60 | TrainLoss 0.3969 | ValLoss 0.3779 | ValMetric 0.8333 | BestTest 0.7500
Epoch  70 | TrainLoss 0.4072 | ValLoss 0.3654 | ValMetric 0.8333 | BestTest 0.7500
Epoch  80 | TrainLoss 0.3456 | ValLoss 0.3515 | ValMetric 0.8333 | BestTest 0.7500
Epoch  90 | TrainLoss 0.3709 | ValLoss 0.3489 | ValMetric 0.8333 | BestTest 0.7500
Epoch 100 | TrainLoss 0.3269 | ValLoss 0.3575 | ValMetric 0.8889 | BestTest 0.8000
[Full-batch] Best epoch 95, Test metric 0.8000, Total time 1.3s
{
  "best_val": 0.8888888888888888,
  "best_test": 0.8,
  "best_epoch": 95,
  "total_time": 1.3475573062896729
}
>>> ...\.conda\envs\globalwork\python.exe train_batch.py --dataset MUTAG --model GIN --pool max --epochs 100 --hidden 64 --layers 3 --lr 0.001 --batch_size 32
[Mini-batch] Device: cuda
Epoch   1 | TrainLoss 0.6929 | ValLoss 0.6850 | ValMetric 0.7778 | BestTest 0.8500
Epoch  10 | TrainLoss 0.4602 | ValLoss 0.3537 | ValMetric 0.9444 | BestTest 0.9000
Epoch  20 | TrainLoss 0.4200 | ValLoss 0.2915 | ValMetric 0.8333 | BestTest 0.9000
Epoch  30 | TrainLoss 0.4582 | ValLoss 0.3142 | ValMetric 0.8889 | BestTest 0.9000
Epoch  40 | TrainLoss 0.3722 | ValLoss 0.3249 | ValMetric 0.8889 | BestTest 0.9000
Epoch  50 | TrainLoss 0.3769 | ValLoss 0.2547 | ValMetric 0.9444 | BestTest 0.9000
Epoch  60 | TrainLoss 0.3565 | ValLoss 0.3798 | ValMetric 0.7778 | BestTest 0.9000
Epoch  70 | TrainLoss 0.3522 | ValLoss 0.2697 | ValMetric 0.8333 | BestTest 0.9000
Epoch  80 | TrainLoss 0.3791 | ValLoss 0.2758 | ValMetric 0.9444 | BestTest 0.9000
Epoch  90 | TrainLoss 0.3213 | ValLoss 0.3549 | ValMetric 0.8333 | BestTest 0.9000
Epoch 100 | TrainLoss 0.3422 | ValLoss 0.2976 | ValMetric 0.8333 | BestTest 0.9000
[Mini-batch] Best epoch 9, Test metric 0.9000, Total time 3.2s
{
  "best_val": 0.9444444444444444,
  "best_test": 0.9,
  "best_epoch": 9,
  "total_time": 3.1726887226104736
}
>>> ...\.conda\envs\globalwork\python.exe train_full.py --dataset MUTAG --model GIN --pool min --epochs 100 --hidden 64 --layers 3 --lr 0.001
[Full-batch] Device: cuda
Epoch   1 | TrainLoss 0.6877 | ValLoss 0.6862 | ValMetric 0.6111 | BestTest 0.4500
Epoch  10 | TrainLoss 0.6745 | ValLoss 0.6788 | ValMetric 0.6111 | BestTest 0.4500
Epoch  20 | TrainLoss 0.6622 | ValLoss 0.6718 | ValMetric 0.6111 | BestTest 0.4500
Epoch  30 | TrainLoss 0.6518 | ValLoss 0.6706 | ValMetric 0.6111 | BestTest 0.4500
Epoch  40 | TrainLoss 0.6368 | ValLoss 0.6715 | ValMetric 0.6111 | BestTest 0.4500
Epoch  50 | TrainLoss 0.6258 | ValLoss 0.6731 | ValMetric 0.6111 | BestTest 0.4500
Epoch  60 | TrainLoss 0.6243 | ValLoss 0.6692 | ValMetric 0.6111 | BestTest 0.4500
Epoch  70 | TrainLoss 0.6177 | ValLoss 0.6724 | ValMetric 0.6111 | BestTest 0.4500
Epoch  80 | TrainLoss 0.6186 | ValLoss 0.6723 | ValMetric 0.6111 | BestTest 0.4500
Epoch  90 | TrainLoss 0.6122 | ValLoss 0.6733 | ValMetric 0.6111 | BestTest 0.4500
Epoch 100 | TrainLoss 0.6067 | ValLoss 0.6738 | ValMetric 0.6111 | BestTest 0.4500
[Full-batch] Best epoch 1, Test metric 0.4500, Total time 1.4s
{
  "best_val": 0.6111111111111112,
  "best_test": 0.45,
  "best_epoch": 1,
  "total_time": 1.3501808643341064
}
>>> ...\.conda\envs\globalwork\python.exe train_batch.py --dataset MUTAG --model GIN --pool min --epochs 100 --hidden 64 --layers 3 --lr 0.001 --batch_size 32
[Mini-batch] Device: cuda
Epoch   1 | TrainLoss 0.6914 | ValLoss 0.6725 | ValMetric 0.7778 | BestTest 0.6000
Epoch  10 | TrainLoss 0.6559 | ValLoss 0.6019 | ValMetric 0.7778 | BestTest 0.6000
Epoch  20 | TrainLoss 0.6426 | ValLoss 0.5737 | ValMetric 0.7778 | BestTest 0.6000
Epoch  30 | TrainLoss 0.6405 | ValLoss 0.5644 | ValMetric 0.7778 | BestTest 0.6000
Epoch  40 | TrainLoss 0.6401 | ValLoss 0.5649 | ValMetric 0.7778 | BestTest 0.6000
Epoch  50 | TrainLoss 0.6447 | ValLoss 0.5650 | ValMetric 0.7778 | BestTest 0.6000
Epoch  60 | TrainLoss 0.6359 | ValLoss 0.5636 | ValMetric 0.7778 | BestTest 0.6000
Epoch  70 | TrainLoss 0.6399 | ValLoss 0.5634 | ValMetric 0.7778 | BestTest 0.6000
Epoch  80 | TrainLoss 0.6360 | ValLoss 0.5633 | ValMetric 0.7778 | BestTest 0.6000
Epoch  90 | TrainLoss 0.6436 | ValLoss 0.5644 | ValMetric 0.7778 | BestTest 0.6000
Epoch 100 | TrainLoss 0.6437 | ValLoss 0.5644 | ValMetric 0.7778 | BestTest 0.6000
[Mini-batch] Best epoch 1, Test metric 0.6000, Total time 3.2s
{
  "best_val": 0.7777777777777778,
  "best_test": 0.6,
  "best_epoch": 1,
  "total_time": 3.2481720447540283
}
[Full-batch] Device: cuda
Epoch   1 | TrainLoss 0.6777 | ValLoss 0.6426 | ValMetric 0.8333 | BestTest 0.6000
Epoch  10 | TrainLoss 0.6598 | ValLoss 0.6466 | ValMetric 0.8333 | BestTest 0.6000
Epoch  20 | TrainLoss 0.6508 | ValLoss 0.6551 | ValMetric 0.8333 | BestTest 0.6000
Epoch  30 | TrainLoss 0.6452 | ValLoss 0.6481 | ValMetric 0.7778 | BestTest 0.6000
Epoch  40 | TrainLoss 0.6154 | ValLoss 0.6286 | ValMetric 0.6111 | BestTest 0.6000
Epoch  50 | TrainLoss 0.5914 | ValLoss 0.6127 | ValMetric 0.5556 | BestTest 0.6000
Epoch  60 | TrainLoss 0.5771 | ValLoss 0.6059 | ValMetric 0.5556 | BestTest 0.6000
Epoch  70 | TrainLoss 0.5895 | ValLoss 0.5951 | ValMetric 0.5556 | BestTest 0.6000
Epoch  80 | TrainLoss 0.5837 | ValLoss 0.5865 | ValMetric 0.5556 | BestTest 0.6000
Epoch  90 | TrainLoss 0.5663 | ValLoss 0.5826 | ValMetric 0.5556 | BestTest 0.6000
Epoch 100 | TrainLoss 0.5562 | ValLoss 0.5812 | ValMetric 0.5556 | BestTest 0.6000
[Full-batch] Best epoch 1, Test metric 0.6000, Total time 1.3s
{
  "best_val": 0.8333333333333334,
  "best_test": 0.6,
  "best_epoch": 1,
  "total_time": 1.3300752639770508
}
[Mini-batch] Device: cuda
Epoch   1 | TrainLoss 0.6700 | ValLoss 0.6742 | ValMetric 0.7222 | BestTest 0.6500
Epoch  10 | TrainLoss 0.6230 | ValLoss 0.6271 | ValMetric 0.7222 | BestTest 0.6500
Epoch  20 | TrainLoss 0.5854 | ValLoss 0.5267 | ValMetric 0.7222 | BestTest 0.6500
Epoch  30 | TrainLoss 0.5273 | ValLoss 0.4676 | ValMetric 0.7222 | BestTest 0.6500
Epoch  40 | TrainLoss 0.4999 | ValLoss 0.4207 | ValMetric 0.8889 | BestTest 0.8500
Epoch  50 | TrainLoss 0.4646 | ValLoss 0.3914 | ValMetric 0.9444 | BestTest 0.9500
Epoch  60 | TrainLoss 0.4574 | ValLoss 0.3784 | ValMetric 0.8889 | BestTest 0.9500
Epoch  70 | TrainLoss 0.4475 | ValLoss 0.3646 | ValMetric 0.8333 | BestTest 0.9500
Epoch  80 | TrainLoss 0.4483 | ValLoss 0.3516 | ValMetric 0.8333 | BestTest 0.9500
Epoch  90 | TrainLoss 0.4310 | ValLoss 0.3346 | ValMetric 0.8889 | BestTest 0.9500
Epoch 100 | TrainLoss 0.4168 | ValLoss 0.3350 | ValMetric 0.8333 | BestTest 0.9500
[Mini-batch] Best epoch 44, Test metric 0.9500, Total time 3.3s
{
  "best_val": 0.9444444444444444,
  "best_test": 0.95,
  "best_epoch": 44,
  "total_time": 3.26842999458313
}
[Full-batch] Device: cuda
Epoch   1 | TrainLoss 0.7014 | ValLoss 0.6613 | ValMetric 0.7222 | BestTest 0.5500
Epoch  10 | TrainLoss 0.5992 | ValLoss 0.6275 | ValMetric 0.7222 | BestTest 0.5500
Epoch  20 | TrainLoss 0.5180 | ValLoss 0.5281 | ValMetric 0.7222 | BestTest 0.5500
Epoch  30 | TrainLoss 0.4214 | ValLoss 0.4667 | ValMetric 0.7222 | BestTest 0.5500
Epoch  40 | TrainLoss 0.4105 | ValLoss 0.4445 | ValMetric 0.7222 | BestTest 0.5500
Epoch  50 | TrainLoss 0.3342 | ValLoss 0.4297 | ValMetric 0.7222 | BestTest 0.5500
Epoch  60 | TrainLoss 0.3659 | ValLoss 0.4519 | ValMetric 0.8333 | BestTest 0.6500
Epoch  70 | TrainLoss 0.3284 | ValLoss 0.2574 | ValMetric 0.8333 | BestTest 0.6500
Epoch  80 | TrainLoss 0.3192 | ValLoss 0.1525 | ValMetric 1.0000 | BestTest 0.8500
Epoch  90 | TrainLoss 0.3060 | ValLoss 0.1586 | ValMetric 0.9444 | BestTest 0.8500
Epoch 100 | TrainLoss 0.2937 | ValLoss 0.1673 | ValMetric 0.9444 | BestTest 0.8500
[Full-batch] Best epoch 76, Test metric 0.8500, Total time 1.4s
{
  "best_val": 1.0,
  "best_test": 0.85,
  "best_epoch": 76,
  "total_time": 1.3615782260894775
}
[Mini-batch] Device: cuda
Epoch   1 | TrainLoss 0.6852 | ValLoss 0.6526 | ValMetric 0.8889 | BestTest 0.6000
Epoch  10 | TrainLoss 0.4063 | ValLoss 0.2611 | ValMetric 0.8333 | BestTest 0.6000
Epoch  20 | TrainLoss 0.3084 | ValLoss 0.2500 | ValMetric 0.8333 | BestTest 0.6000
Epoch  30 | TrainLoss 0.3187 | ValLoss 0.2730 | ValMetric 0.8889 | BestTest 0.7000
Epoch  40 | TrainLoss 0.2852 | ValLoss 0.2413 | ValMetric 0.8333 | BestTest 0.7000
Epoch  50 | TrainLoss 0.2420 | ValLoss 0.3133 | ValMetric 0.8889 | BestTest 0.7000
Epoch  60 | TrainLoss 0.2331 | ValLoss 0.3154 | ValMetric 0.8889 | BestTest 0.7000
Epoch  70 | TrainLoss 0.2304 | ValLoss 0.2690 | ValMetric 0.8333 | BestTest 0.7000
Epoch  80 | TrainLoss 0.2336 | ValLoss 0.3562 | ValMetric 0.8889 | BestTest 0.7000
Epoch  90 | TrainLoss 0.1924 | ValLoss 0.4329 | ValMetric 0.7778 | BestTest 0.7000
Epoch 100 | TrainLoss 0.2308 | ValLoss 0.3782 | ValMetric 0.8333 | BestTest 0.7000
[Mini-batch] Best epoch 25, Test metric 0.7000, Total time 3.2s
{
  "best_val": 0.9444444444444444,
  "best_test": 0.7,
  "best_epoch": 25,
  "total_time": 3.238196611404419
}
[Full-batch] Device: cuda
Epoch   1 | TrainLoss 0.6964 | ValLoss 0.6521 | ValMetric 0.6667 | BestTest 0.7000
Epoch  10 | TrainLoss 0.3587 | ValLoss 12.3745 | ValMetric 0.6667 | BestTest 0.7000
Epoch  20 | TrainLoss 0.3275 | ValLoss 12.3367 | ValMetric 0.6667 | BestTest 0.7000
Epoch  30 | TrainLoss 0.3548 | ValLoss 4.7568 | ValMetric 0.7222 | BestTest 0.7000
Epoch  40 | TrainLoss 0.3138 | ValLoss 0.8498 | ValMetric 0.7222 | BestTest 0.8000
Epoch  50 | TrainLoss 0.2539 | ValLoss 1.6924 | ValMetric 0.7222 | BestTest 0.8000
Epoch  60 | TrainLoss 0.2649 | ValLoss 0.8033 | ValMetric 0.7778 | BestTest 0.8000
Epoch  70 | TrainLoss 0.2276 | ValLoss 0.1739 | ValMetric 0.9444 | BestTest 0.8000
Epoch  80 | TrainLoss 0.2006 | ValLoss 0.1913 | ValMetric 0.8889 | BestTest 0.8000
Epoch  90 | TrainLoss 0.1855 | ValLoss 0.1744 | ValMetric 0.8889 | BestTest 0.8000
Epoch 100 | TrainLoss 0.1853 | ValLoss 0.2709 | ValMetric 0.8889 | BestTest 0.8000
[Full-batch] Best epoch 71, Test metric 0.8000, Total time 1.4s
{
  "best_val": 1.0,
  "best_test": 0.8,
  "best_epoch": 71,
  "total_time": 1.3665978908538818
}
[Mini-batch] Device: cuda
Epoch   1 | TrainLoss 0.5984 | ValLoss 1.3436 | ValMetric 0.7222 | BestTest 0.7000
Epoch  10 | TrainLoss 0.4020 | ValLoss 0.4689 | ValMetric 0.8333 | BestTest 0.7500
Epoch  20 | TrainLoss 0.3445 | ValLoss 0.3544 | ValMetric 0.7778 | BestTest 0.9000
Epoch  30 | TrainLoss 0.3607 | ValLoss 0.5219 | ValMetric 0.7778 | BestTest 0.9000
Epoch  40 | TrainLoss 0.3564 | ValLoss 0.2772 | ValMetric 0.8889 | BestTest 0.9000
Epoch  50 | TrainLoss 0.3257 | ValLoss 0.2296 | ValMetric 0.8889 | BestTest 0.8000
Epoch  60 | TrainLoss 0.2377 | ValLoss 0.3989 | ValMetric 0.7222 | BestTest 0.8000
Epoch  70 | TrainLoss 0.2910 | ValLoss 0.3088 | ValMetric 0.8333 | BestTest 0.8000
Epoch  80 | TrainLoss 0.2568 | ValLoss 0.2394 | ValMetric 0.9444 | BestTest 0.8000
Epoch  90 | TrainLoss 0.2175 | ValLoss 0.1967 | ValMetric 0.9444 | BestTest 0.8000
Epoch 100 | TrainLoss 0.1925 | ValLoss 0.3320 | ValMetric 0.7778 | BestTest 0.8000
[Mini-batch] Best epoch 48, Test metric 0.8000, Total time 3.2s
{
  "best_val": 0.9444444444444444,
  "best_test": 0.8,
  "best_epoch": 48,
  "total_time": 3.2034542560577393
}
[Full-batch] Device: cuda
[Mini-batch] Device: cuda
Epoch   1 | TrainLoss 0.6806 | ValLoss 0.6864 | ValMetric 0.7778 | BestTest 0.9000
Epoch  10 | TrainLoss 0.4051 | ValLoss 0.4352 | ValMetric 0.7778 | BestTest 0.9000
Epoch  20 | TrainLoss 0.4092 | ValLoss 0.3759 | ValMetric 0.7222 | BestTest 0.9000
Epoch  30 | TrainLoss 0.3700 | ValLoss 0.2585 | ValMetric 0.8333 | BestTest 0.9000
Epoch  40 | TrainLoss 0.2929 | ValLoss 0.2787 | ValMetric 0.8889 | BestTest 0.9500
Epoch  50 | TrainLoss 0.3346 | ValLoss 0.2867 | ValMetric 0.8889 | BestTest 0.9500
Epoch  60 | TrainLoss 0.3039 | ValLoss 0.1879 | ValMetric 0.9444 | BestTest 0.9500
Epoch  70 | TrainLoss 0.2861 | ValLoss 0.1773 | ValMetric 0.9444 | BestTest 0.9500
Epoch  80 | TrainLoss 0.3060 | ValLoss 0.1824 | ValMetric 0.9444 | BestTest 0.9500
Epoch  90 | TrainLoss 0.3018 | ValLoss 0.1587 | ValMetric 1.0000 | BestTest 0.9500
Epoch 100 | TrainLoss 0.2853 | ValLoss 0.1459 | ValMetric 1.0000 | BestTest 0.9500
[Mini-batch] Best epoch 36, Test metric 0.9500, Total time 3.0s
{
  "best_val": 1.0,
  "best_test": 0.95,
  "best_epoch": 36,
  "total_time": 2.985283613204956
}
[Full-batch] Device: cuda
Epoch   1 | TrainLoss 0.6758 | ValLoss 0.6707 | ValMetric 0.7778 | BestTest 0.6500
Epoch  10 | TrainLoss 0.5374 | ValLoss 0.5669 | ValMetric 0.7778 | BestTest 0.6500
Epoch  20 | TrainLoss 0.4520 | ValLoss 0.4052 | ValMetric 0.7778 | BestTest 0.6500
Epoch  30 | TrainLoss 0.3907 | ValLoss 0.3973 | ValMetric 0.7778 | BestTest 0.6500
Epoch  40 | TrainLoss 0.3945 | ValLoss 0.3911 | ValMetric 0.7222 | BestTest 0.6500
Epoch  50 | TrainLoss 0.3229 | ValLoss 0.3402 | ValMetric 0.6667 | BestTest 0.6500
Epoch  60 | TrainLoss 0.3246 | ValLoss 0.3433 | ValMetric 0.7222 | BestTest 0.6500
Epoch  70 | TrainLoss 0.3129 | ValLoss 0.3256 | ValMetric 0.8333 | BestTest 0.9000
Epoch  80 | TrainLoss 0.2916 | ValLoss 0.3052 | ValMetric 0.8333 | BestTest 0.9000
Epoch  90 | TrainLoss 0.2683 | ValLoss 0.3691 | ValMetric 0.8333 | BestTest 0.9000
Epoch 100 | TrainLoss 0.2873 | ValLoss 0.3059 | ValMetric 0.8333 | BestTest 0.9000
[Full-batch] Best epoch 70, Test metric 0.9000, Total time 1.3s
{
  "best_val": 0.8333333333333334,
  "best_test": 0.9,
  "best_epoch": 70,
  "total_time": 1.345872163772583
}
[Mini-batch] Device: cuda
Epoch   1 | TrainLoss 0.6668 | ValLoss 0.6775 | ValMetric 0.7222 | BestTest 0.6000
Epoch  10 | TrainLoss 0.3907 | ValLoss 0.3093 | ValMetric 0.8889 | BestTest 0.6500
Epoch  20 | TrainLoss 0.3627 | ValLoss 0.3932 | ValMetric 0.7778 | BestTest 0.6500
Epoch  30 | TrainLoss 0.3431 | ValLoss 0.2828 | ValMetric 0.8889 | BestTest 0.6500
Epoch  40 | TrainLoss 0.3078 | ValLoss 0.2729 | ValMetric 0.8333 | BestTest 0.6500
Epoch  50 | TrainLoss 0.3037 | ValLoss 0.1989 | ValMetric 0.9444 | BestTest 0.7000
Epoch  60 | TrainLoss 0.2479 | ValLoss 0.2588 | ValMetric 0.8333 | BestTest 0.7000
Epoch  70 | TrainLoss 0.2721 | ValLoss 0.2758 | ValMetric 0.8333 | BestTest 0.7000
Epoch  80 | TrainLoss 0.2455 | ValLoss 0.2624 | ValMetric 0.8333 | BestTest 0.7000
Epoch  90 | TrainLoss 0.2297 | ValLoss 0.2693 | ValMetric 0.8333 | BestTest 0.7000
Epoch 100 | TrainLoss 0.2442 | ValLoss 0.2449 | ValMetric 0.8889 | BestTest 0.7000
[Mini-batch] Best epoch 50, Test metric 0.7000, Total time 3.2s
{
  "best_val": 0.9444444444444444,
  "best_test": 0.7,
  "best_epoch": 50,
  "total_time": 3.163490056991577
}
[Full-batch] Device: cuda
Epoch   1 | TrainLoss 0.7383 | ValLoss 0.7153 | ValMetric 0.2222 | BestTest 0.3500
Epoch  10 | TrainLoss 0.6118 | ValLoss 0.6014 | ValMetric 0.7778 | BestTest 0.6500
Epoch  20 | TrainLoss 0.4827 | ValLoss 0.4243 | ValMetric 0.7778 | BestTest 0.6500
Epoch  30 | TrainLoss 0.4464 | ValLoss 0.5078 | ValMetric 0.8333 | BestTest 0.6500
Epoch  40 | TrainLoss 0.3644 | ValLoss 0.4211 | ValMetric 0.8889 | BestTest 0.7500
Epoch  50 | TrainLoss 0.3020 | ValLoss 0.3379 | ValMetric 0.8889 | BestTest 0.7500
Epoch  60 | TrainLoss 0.3137 | ValLoss 0.1412 | ValMetric 0.9444 | BestTest 0.6500
Epoch  70 | TrainLoss 0.3062 | ValLoss 0.1367 | ValMetric 0.9444 | BestTest 0.8000
Epoch  80 | TrainLoss 0.2534 | ValLoss 0.1256 | ValMetric 0.9444 | BestTest 0.8000
Epoch  90 | TrainLoss 0.2806 | ValLoss 0.1226 | ValMetric 1.0000 | BestTest 0.8000
Epoch 100 | TrainLoss 0.2511 | ValLoss 0.1424 | ValMetric 0.9444 | BestTest 0.8000
[Full-batch] Best epoch 65, Test metric 0.8000, Total time 1.5s
{
  "best_val": 1.0,
  "best_test": 0.8,
  "best_epoch": 65,
  "total_time": 1.4723198413848877
}
[Mini-batch] Device: cuda
Epoch   1 | TrainLoss 0.7057 | ValLoss 0.7124 | ValMetric 0.2778 | BestTest 0.5000
Epoch  10 | TrainLoss 0.4425 | ValLoss 0.2560 | ValMetric 0.8889 | BestTest 0.7000
Epoch  20 | TrainLoss 0.3902 | ValLoss 0.1831 | ValMetric 0.8889 | BestTest 0.7000
Epoch  30 | TrainLoss 0.4194 | ValLoss 0.1848 | ValMetric 0.9444 | BestTest 0.7500
Epoch  40 | TrainLoss 0.2988 | ValLoss 0.2249 | ValMetric 0.9444 | BestTest 0.7500
Epoch  50 | TrainLoss 0.3146 | ValLoss 0.1390 | ValMetric 0.9444 | BestTest 0.7500
Epoch  60 | TrainLoss 0.2834 | ValLoss 0.0961 | ValMetric 0.9444 | BestTest 0.7500
Epoch  70 | TrainLoss 0.2734 | ValLoss 0.1355 | ValMetric 0.9444 | BestTest 0.7500
Epoch  80 | TrainLoss 0.2744 | ValLoss 0.1239 | ValMetric 0.9444 | BestTest 0.9000
Epoch  90 | TrainLoss 0.3000 | ValLoss 0.4242 | ValMetric 0.7778 | BestTest 0.9000
Epoch 100 | TrainLoss 0.2913 | ValLoss 0.1526 | ValMetric 0.9444 | BestTest 0.9000
[Mini-batch] Best epoch 73, Test metric 0.9000, Total time 3.5s
{
  "best_val": 1.0,
  "best_test": 0.9,
  "best_epoch": 73,
  "total_time": 3.538600206375122
}
[Mini-batch] Device: cuda
Epoch   1 | TrainLoss 0.6254 | ValLoss 0.6509 | ValMetric 0.8333 | BestTest 0.6500
Epoch  10 | TrainLoss 0.3675 | ValLoss 0.3717 | ValMetric 0.7778 | BestTest 0.6500
Epoch  20 | TrainLoss 0.4071 | ValLoss 0.3771 | ValMetric 0.7778 | BestTest 0.6500
Epoch  30 | TrainLoss 0.3455 | ValLoss 0.3872 | ValMetric 0.7778 | BestTest 0.6500
Epoch  40 | TrainLoss 0.2719 | ValLoss 0.3496 | ValMetric 0.7778 | BestTest 0.6500
Epoch  50 | TrainLoss 0.3719 | ValLoss 0.4562 | ValMetric 0.7778 | BestTest 0.6500
Epoch  60 | TrainLoss 0.2704 | ValLoss 0.4420 | ValMetric 0.7778 | BestTest 0.6500
Epoch  70 | TrainLoss 0.2614 | ValLoss 0.4619 | ValMetric 0.7778 | BestTest 0.6500
Epoch  80 | TrainLoss 0.2171 | ValLoss 0.4123 | ValMetric 0.7778 | BestTest 0.6500
Epoch  90 | TrainLoss 0.3294 | ValLoss 0.3926 | ValMetric 0.7778 | BestTest 0.6500
Epoch 100 | TrainLoss 0.2622 | ValLoss 0.5137 | ValMetric 0.7778 | BestTest 0.6500
[Mini-batch] Best epoch 1, Test metric 0.6500, Total time 5.6s
{
  "best_val": 0.8333333333333334,
  "best_test": 0.65,
  "best_epoch": 1,
  "total_time": 5.591905117034912
}
[Mini-batch] Device: cuda
Epoch   1 | TrainLoss 0.6493 | ValLoss 0.6975 | ValMetric 0.5000 | BestTest 0.6500
Epoch  10 | TrainLoss 0.3872 | ValLoss 0.6014 | ValMetric 0.6111 | BestTest 0.8000
Epoch  20 | TrainLoss 0.3363 | ValLoss 0.4156 | ValMetric 0.7222 | BestTest 0.9000
Epoch  30 | TrainLoss 0.2919 | ValLoss 0.4660 | ValMetric 0.7222 | BestTest 0.9000
Epoch  40 | TrainLoss 0.2882 | ValLoss 0.3609 | ValMetric 0.8889 | BestTest 0.9000
Epoch  50 | TrainLoss 0.3128 | ValLoss 1.0590 | ValMetric 0.7222 | BestTest 0.9000
Epoch  60 | TrainLoss 0.2567 | ValLoss 0.3170 | ValMetric 0.8889 | BestTest 0.9000
Epoch  70 | TrainLoss 0.2297 | ValLoss 0.3194 | ValMetric 0.8333 | BestTest 0.9000
Epoch  80 | TrainLoss 0.2297 | ValLoss 0.3130 | ValMetric 0.8889 | BestTest 0.9000
Epoch  90 | TrainLoss 0.2457 | ValLoss 0.2927 | ValMetric 0.8889 | BestTest 0.9000
Epoch 100 | TrainLoss 0.3212 | ValLoss 0.2816 | ValMetric 0.8889 | BestTest 0.9000
[Mini-batch] Best epoch 21, Test metric 0.9000, Total time 3.1s
{
  "best_val": 0.8888888888888888,
  "best_test": 0.9,
  "best_epoch": 21,
  "total_time": 3.1486642360687256
}
[Mini-batch] Device: cuda
Epoch   1 | TrainLoss 0.6531 | ValLoss 0.6654 | ValMetric 0.7778 | BestTest 0.7500
Epoch  10 | TrainLoss 0.4473 | ValLoss 0.4169 | ValMetric 0.7778 | BestTest 0.7500
Epoch  20 | TrainLoss 0.3891 | ValLoss 0.3965 | ValMetric 0.7222 | BestTest 0.7500
Epoch  30 | TrainLoss 0.3273 | ValLoss 0.4618 | ValMetric 0.7778 | BestTest 0.7500
Epoch  40 | TrainLoss 0.3572 | ValLoss 0.5590 | ValMetric 0.7778 | BestTest 0.7500
Epoch  50 | TrainLoss 0.2980 | ValLoss 0.6147 | ValMetric 0.7778 | BestTest 0.7500
Epoch  60 | TrainLoss 0.2552 | ValLoss 0.4802 | ValMetric 0.7778 | BestTest 0.7500
Epoch  70 | TrainLoss 0.3279 | ValLoss 0.4374 | ValMetric 0.8333 | BestTest 0.8000
Epoch  80 | TrainLoss 0.2560 | ValLoss 0.4170 | ValMetric 0.8333 | BestTest 0.8000
Epoch  90 | TrainLoss 0.2773 | ValLoss 0.4467 | ValMetric 0.7778 | BestTest 0.7000
Epoch 100 | TrainLoss 0.2681 | ValLoss 0.5028 | ValMetric 0.7778 | BestTest 0.7000
[Mini-batch] Best epoch 86, Test metric 0.7000, Total time 2.2s
{
  "best_val": 0.8888888888888888,
  "best_test": 0.7,
  "best_epoch": 86,
  "total_time": 2.221301555633545
}
[Mini-batch] Device: cuda
Epoch   1 | TrainLoss 0.6940 | ValLoss 0.6912 | ValMetric 0.7778 | BestTest 0.8000
Epoch  10 | TrainLoss 0.5377 | ValLoss 0.5943 | ValMetric 0.7778 | BestTest 0.8000
Epoch  20 | TrainLoss 0.4370 | ValLoss 0.3738 | ValMetric 0.8333 | BestTest 0.8000
Epoch  30 | TrainLoss 0.3910 | ValLoss 0.2869 | ValMetric 0.7778 | BestTest 0.8000
Epoch  40 | TrainLoss 0.4227 | ValLoss 0.2820 | ValMetric 0.8889 | BestTest 0.8000
Epoch  50 | TrainLoss 0.3683 | ValLoss 0.2903 | ValMetric 0.8333 | BestTest 0.8000
Epoch  60 | TrainLoss 0.3768 | ValLoss 0.2078 | ValMetric 0.8333 | BestTest 0.8000
Epoch  70 | TrainLoss 0.3362 | ValLoss 0.2908 | ValMetric 0.7778 | BestTest 0.8000
Epoch  80 | TrainLoss 0.3172 | ValLoss 0.3424 | ValMetric 0.7222 | BestTest 0.8000
Epoch  90 | TrainLoss 0.3127 | ValLoss 0.2728 | ValMetric 0.8333 | BestTest 0.8000
Epoch 100 | TrainLoss 0.3286 | ValLoss 0.3475 | ValMetric 0.8333 | BestTest 0.8000
[Mini-batch] Best epoch 2, Test metric 0.8000, Total time 1.8s
{
  "best_val": 0.8888888888888888,
  "best_test": 0.8,
  "best_epoch": 2,
  "total_time": 1.801008939743042
}

All experiments finished.
