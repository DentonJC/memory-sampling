guild run main.py batch_size=32 batch_size_mem=32 model=cnn dataset=[cifar10] memory_size=[2000] lr=[0.005] weight_decay=0.0 epochs=1 strategy=[er] trials=20 vectors=1  online=True --yes &
guild run main.py batch_size=32 batch_size_mem=32 model=cnn dataset=[cifar10] memory_size=[2000] lr=[0.005] weight_decay=0.0 epochs=1 strategy=[nu] trials=20 vectors=50 online=True seed=2 --yes &

guild run main.py batch_size=32 batch_size_mem=32 model=resnet18 dataset=[cifar10] memory_size=[2000] lr=[0.0005] weight_decay=0.0 epochs=1 strategy=[er] trials=20 vectors=1  online=True --yes &
guild run main.py batch_size=32 batch_size_mem=32 model=resnet18 dataset=[cifar10] memory_size=[2000] lr=[0.0005] weight_decay=0.0 epochs=1 strategy=[nu] trials=20 vectors=50 online=True seed=2 --yes &
wait
