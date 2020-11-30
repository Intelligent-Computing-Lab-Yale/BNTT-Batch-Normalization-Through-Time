#############################################
#   @author: Youngeun Kim and Priya Panda   #
#############################################
#--------------------------------------------------
# Imports
#--------------------------------------------------
import torch.optim as optim
import torchvision
from   torch.utils.data.dataloader import DataLoader
from   torchvision import transforms
from   model import *

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse
import os.path
import numpy as np
import torch.backends.cudnn as cudnn
from utills import *

cudnn.benchmark = True
cudnn.deterministic = True

#--------------------------------------------------
# Parse input arguments
#--------------------------------------------------
parser = argparse.ArgumentParser(description='SNN trained with BNTT', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--seed',                  default=0,        type=int,   help='Random seed')
parser.add_argument('--num_steps',             default=25,    type=int, help='Number of time-step')
parser.add_argument('--batch_size',            default=128,       type=int,   help='Batch size')
parser.add_argument('--lr',                    default=0.1,   type=float, help='Learning rate')
parser.add_argument('--leak_mem',              default=0.99,   type=float, help='Leak_mem')
parser.add_argument('--arch',              default='vgg11',   type=str, help='Dataset [vgg9, vgg11]')
parser.add_argument('--dataset',              default='cifar100',   type=str, help='Dataset [cifar10, cifar100]')
parser.add_argument('--num_epochs',            default=120,       type=int,   help='Number of epochs')
parser.add_argument('--num_workers',           default=4, type=int, help='number of workers')
parser.add_argument('--train_display_freq',    default=10, type=int, help='display_freq for train')
parser.add_argument('--test_display_freq',     default=10, type=int, help='display_freq for test')


global args
args = parser.parse_args()


#--------------------------------------------------
# Initialize tensorboard setting
#--------------------------------------------------
log_dir = 'modelsave'
if os.path.isdir(log_dir) is not True:
    os.mkdir(log_dir)


user_foldername = (args.dataset)+(args.arch)+'_timestep'+str(args.num_steps) +'_lr'+str(args.lr) + '_epoch' + str(args.num_epochs) + '_leak' + str(args.leak_mem)



#--------------------------------------------------
# Initialize seed
#--------------------------------------------------
seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

#--------------------------------------------------
# SNN configuration parameters
#--------------------------------------------------
# Leaky-Integrate-and-Fire (LIF) neuron parameters
leak_mem = args.leak_mem

# SNN learning and evaluation parameters
batch_size      = args.batch_size
batch_size_test = args.batch_size*2
num_epochs      = args.num_epochs
num_steps       = args.num_steps
lr   = args.lr


#--------------------------------------------------
# Load  dataset
#--------------------------------------------------

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


if args.dataset == 'cifar10':
    num_cls = 10
    img_size = 32

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform_test)
elif args.dataset == 'cifar100':
    num_cls = 100
    img_size = 32

    train_set = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR100(root='./data', train=False,
                                            download=True, transform=transform_test)
else:
    print("not implemented yet..")
    exit()



trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
testloader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)



#--------------------------------------------------
# Instantiate the SNN model and optimizer
#--------------------------------------------------
if args.arch == 'vgg9':
    model = SNN_VGG9_BNTT(num_steps = num_steps, leak_mem=leak_mem, img_size=img_size,  num_cls=num_cls)
elif args.arch == 'vgg11':
    model = SNN_VGG11_BNTT(num_steps = num_steps, leak_mem=leak_mem, img_size=img_size,  num_cls=num_cls)
else:
    print("not implemented yet..")
    exit()

model = model.cuda()

# Configure the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr,momentum=0.9,weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
best_acc = 0

# Print the SNN model, optimizer, and simulation parameters
print('********** SNN simulation parameters **********')
print('Simulation # time-step : {}'.format(num_steps))
print('Membrane decay rate : {0:.2f}\n'.format(leak_mem))

print('********** SNN learning parameters **********')
print('Backprop optimizer     : SGD')
print('Batch size (training)  : {}'.format(batch_size))
print('Batch size (testing)   : {}'.format(batch_size_test))
print('Number of epochs       : {}'.format(num_epochs))
print('Learning rate          : {}'.format(lr))

#--------------------------------------------------
# Train the SNN using surrogate gradients
#--------------------------------------------------
print('********** SNN training and evaluation **********')
train_loss_list = []
test_acc_list = []

for epoch in range(num_epochs):
    train_loss = AverageMeter()
    model.train()
    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        output = model(inputs)

        loss   = criterion(output, labels)

        prec1, prec5 = accuracy(output, labels, topk=(1, 5))
        train_loss.update(loss.item(), labels.size(0))

        loss.backward()
        optimizer.step()

    if (epoch+1) % args.train_display_freq ==0:
        print("Epoch: {}/{};".format(epoch+1, num_epochs), "########## Training loss: {}".format(train_loss.avg))

    adjust_learning_rate(optimizer, epoch, num_epochs)



    if (epoch+1) %  args.test_display_freq ==0:
        acc_top1, acc_top5 = [], []
        model.eval()
        with torch.no_grad():
             for j, data in enumerate(testloader, 0):

                 images, labels = data
                 images = images.cuda()
                 labels = labels.cuda()

                 out = model(images)
                 prec1, prec5 = accuracy(out, labels, topk=(1, 5))
                 acc_top1.append(float(prec1))
                 acc_top5.append(float(prec5))


        test_accuracy = np.mean(acc_top1)
        print ("test_accuracy : {}". format(test_accuracy))


        # Model save
        if best_acc < test_accuracy:
            best_acc = test_accuracy

            model_dict = {
                    'global_step': epoch + 1,
                    'state_dict': model.state_dict(),
                    'accuracy': test_accuracy}

            torch.save(model_dict, log_dir+'/'+user_foldername+'_bestmodel.pth.tar')


sys.exit(0)


