import time

class Model(nn.Module):
    def __init__(self, num_classes=10):
        super(Model, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out



def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-n','--nodes',default=1,type=int,metavar='N')
  parser.add_argument('-g','--gpus',default=1,type=int,help="number of gpus per node")
  parser.add_argument('-nr','--nr',default=0,type=int,help='ranking within nodes')
  parser.add_argument('--epochs',default=2,type=int,metavar='N')

  args = parser.parse_args()

  args.world_size = args.gpus * args.nodes
  os.environ['MASTER_ADDR'] = '10.57.23.164'
  os.environ['MASTER_PORT'] = '8888'
  mp.spawn(train,nprocs=args.gpus,args=(args,))


###################################



def train(gpu,args):

  rank = args.nr * args.gpus + gpu
  dist.init_process_group(
      backend='nccl',
      init_method = 'env://',
      world_size = args.world_size,
      rank = rank
  )
  #######################
  model = Model()
  torch.cuda.set_device(gpu)
  model.cuda(gpu)

  batch_size = 100

  criterion = nn.CrossEntropyLoss().cuda(gpu)
  optim = torch.optim.SGD(model.parameters(),1e-4)

  model = nn.parallel.DistributedDataParallel(model,device_ids=[gpu])

  ##############

  train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
  train_sampler = torch.utils.data.distributed.DistributedSampler(
      train_dataset,
      num_replicas = args.world_size,
      rank = rank
  )
  train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=0,
                                               pin_memory=True,
                                             sampler = train_sampler)
  start = time.time()

  total_step = len(train_loader)

  for epoch in range(args.epochs):
    for i,(images,labels) in enumerate(train_loader):
      images = images.cuda(non_blocking=True)
      labels = None
      outputs = model(images)

      loss = criterion(outputs, labels)
      #zero grad
      #loss backward 

      #optim

      if gpu==0:
        print(f"Epoch:{epoch+1}/{args.epochs},Loss:{loss.item()}")
  if gpu == 0:
    print(f"training completed in {str(time.time()-start)}")

